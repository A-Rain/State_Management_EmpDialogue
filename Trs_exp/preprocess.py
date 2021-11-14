import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import os
import json
import torch
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
from typing import List, Dict
import argparse


word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
              "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
              "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
              "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not", "hadn't": "had not",
              "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are",
              "i'll": "i will", "wouldn't": "would not", "haven't": "have not", "hasn't": "has not"}



def tokenize_sent(sentence: str, clean_pair=True, lower=True):
    if lower:
        sentence = sentence.lower()
    if clean_pair:
        for k, v in word_pairs.items():
            sentence = sentence.replace(k,v)
    sentence = word_tokenize(sentence)
    return sentence


def get_pretrained_embedding_weights(dialog_json_dir: str, glove_embedding_path: str, save_dir: str, embed_dim=300):
    whole_vocab = set()

    for d_type in ['train', 'valid', 'test']:
        file_name = dialog_json_dir.format(d_type)
        with open(file_name, 'r', encoding='utf-8') as f:
            whole_data = json.load(f)
            for element in tqdm(whole_data, desc=f'filter {d_type}'):
                for utte in element['utterence']:
                    utterence_tokenize = tokenize_sent(utte)
                    whole_vocab.update(utterence_tokenize)
    

    # read glove embedding and build new vocab
    new_vocab = {'<pad>': 0, '<unk>': 1, '<eos0>': 2, '<eos1>': 3, '<bos>': 4, '<cls>': 5, "the": 6}
    vocab_weights = np.zeros((1, embed_dim))
    vocab_weights = np.append(vocab_weights, np.random.randn(6, embed_dim), axis=0)
    idx = 7
    with open(glove_embedding_path, 'r', encoding='utf-8') as f:
        _ = f.readline()
        for line in tqdm(f, desc='read glove'):
            line = line.split()
            if len(line) != embed_dim + 1:
                continue
            word, vec = line[0], [float(i) for i in line[1:]]
            if word in whole_vocab:
                new_vocab.update({word: idx})
                vocab_weights = np.append(vocab_weights, np.array([vec]), axis=0)
                idx += 1
    
    # calculate OOV
    print("vocab size: ", len(new_vocab))
    print("cover rate: ", (len(new_vocab) - 6) / len(whole_vocab))

    # save
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(new_vocab, f)
    np.save(os.path.join(save_dir, 'embed_weights'), vocab_weights)


class Tokenizer(object):
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = dict([(_id, _word) for _word, _id in self.word_to_id.items()])

        self.vocab_size = len(self.word_to_id)

        self.cls_token = '<cls>'
        self.cls_token_id = self.word_to_id['<cls>']

        self.bos_token = '<bos>'
        self.bos_token_id = self.word_to_id['<bos>']

        self.pad_token = '<pad>'
        self.pad_token_id = self.word_to_id['<pad>']

        self.unk_token = '<unk>'
        self.unk_token_id = self.word_to_id['<unk>']

        self.eos0_token = '<eos0>'
        self.eos0_token_id = self.word_to_id['<eos0>']
        self.eos1_token = '<eos1>'
        self.eos1_token_id = self.word_to_id['<eos1>']

        self.special_token_ids = [self.cls_token_id, self.bos_token_id, self.pad_token_id,
                                  self.eos0_token_id, self.eos1_token_id]

    def tokenize(self, sentence: str, clean_pair=True, lower=True):
        if lower:
            sentence = sentence.lower()
        if clean_pair:
            for k, v in word_pairs.items():
                sentence = sentence.replace(k,v)
        sentence = word_tokenize(sentence)
        return sentence

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.word_to_id[token] if token in self.word_to_id else self.unk_token_id for token in tokens]
        else:
            if tokens in self.word_to_id:
                return self.word_to_id[tokens]
            else:
                return self.unk_token_id

    def encode(self, sentence: str, clean_pair=True, lower=True):
        tokens = self.tokenize(sentence, clean_pair, lower)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, word_ids: List[int], skip_special_tokens=True):
        if skip_special_tokens:
            words = [self.id_to_word[_id] for _id in word_ids if _id not in self.special_token_ids]
        else:
            words = [self.id_to_word[_id] for _id in word_ids]
        return " ".join(words)


class Input_feature(object):
    def __init__(self, input_ids_NLG, type_id, input_masks_NLG, label_id, cls_mask, history_mask, 
                 input_ids_NLU, input_masks_NLU, curr_emotion_ids, fut_emotion_ids, intent_label):
        # model embedding input
        self.input_id_NLG = input_ids_NLG
        self.input_mask_NLG = input_masks_NLG
        self.type_id = type_id
        self.cls_mask = cls_mask
        self.history_mask = history_mask
        self.label_id = label_id

        self.input_ids_NLU = input_ids_NLU
        self.input_masks_NLU = input_masks_NLU
        self.curr_emotion_ids = curr_emotion_ids
        self.fut_emotion_ids = fut_emotion_ids
        self.intent_label = intent_label


def _truncate_seq_pair(tokens_hist, tokens_resp, max_length, forward_truncate=False):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_hist) + len(tokens_resp)
        if total_length <= max_length:
            break
        if len(tokens_hist) > len(tokens_resp):
            if forward_truncate:
                tokens_hist = tokens_hist[1:]
            else:
                tokens_hist = tokens_hist[:-1]
        else:
            tokens_resp = tokens_resp[:-1]
    return tokens_hist, tokens_resp


def build_training_data(cfg, tokenizer: Tokenizer, input_json_file, output_cache_file, data_type, 
                        label_tag_to_emo: Dict[str, int], label_tag_to_itt: Dict[str, int], forward_truncate=False,
                        max_seq_length=128, speaker0_state=1, speaker1_state=2, 
                       speaker_state_pad=0, emotion_pad=0, speaker0_token='<eos0>', speaker1_token='<eos1>'):
    
    features = []
    with open(input_json_file, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for data in tqdm(whole_data, desc=f'build {data_type} feature'):
            utterence, token_type_ids = [], []
            intent, emotion = data['intents'], data['emotions']

            for idx, sent in enumerate(data['utterence']):
                sent_token = tokenizer.tokenize(sent) + [speaker0_token if idx % 2 == 0 else speaker1_token]
                token_type_ids.append([speaker0_state if idx % 2 == 0 else speaker1_state] * len(sent_token))
                utterence.append(sent_token)
            
            # build feature
            for i in range(1, len(utterence)):
                if i % 2 == 0:
                    continue

                # build input for NLU
                input_ids_NLU = [tokenizer.cls_token_id]
                for no in range(i):
                    input_ids_NLU.extend(tokenizer.encode(data['utterence'][no]))
                input_ids_NLU = input_ids_NLU[:max_seq_length]
                pad_length = max_seq_length - len(input_ids_NLU)
                attention_masks_NLU = [1] * len(input_ids_NLU) + [0] * pad_length
                input_ids_NLU = input_ids_NLU + [tokenizer.pad_token_id] * pad_length

                # get context token, current_emotion and token type id
                context = [token for sent_token in utterence[:i] for token in sent_token]
                type_ids = [type_id for sent_type in token_type_ids[:i] for type_id in sent_type]
                context_emotion = label_tag_to_emo[emotion[i-1]]

                # get response token and response emotion id
                response, response_emotion = utterence[i], label_tag_to_emo[emotion[i]]

                # build intent label
                intent_label = [0] * len(label_tag_to_itt)
                for itt in intent[i]:
                    intent_label[label_tag_to_itt[itt]] = 1

                # truncate context and response together, -2 counts for cls_token, bos_token
                context, response = _truncate_seq_pair(context, response, max_seq_length - 2, forward_truncate=forward_truncate)
                truncate_context_len, truncate_resp_len = len(context), len(response)          

                if forward_truncate:
                    type_ids = [speaker_state_pad] + type_ids[-truncate_context_len:] + [speaker1_state] * (1 + truncate_resp_len)
                else:
                    type_ids = [speaker_state_pad] + type_ids[:truncate_context_len] + [speaker1_state] * (1 + truncate_resp_len)
    
                 # do not forget to add cls_token in the beginning
                input_ids_NLG = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + context + [tokenizer.bos_token] + response)

                # pad whole to max_seq_length
                pad_length = max_seq_length - len(input_ids_NLG)
                input_ids_NLG = input_ids_NLG + [tokenizer.pad_token_id] * pad_length
                type_ids = type_ids + [speaker_state_pad] * pad_length
                # only look at dialog history
                cls_mask_row = [1] * (truncate_context_len + 1) + [0] *(1+truncate_resp_len) + [0] * pad_length
                cls_mask_col = [1] + [0] * (max_seq_length - 1)
                attention_masks_NLG = [1] * (1 + truncate_context_len + 1 + truncate_resp_len) + [0] * pad_length
                # for label_ids, we need to predict next
                label_ids = [-1] * (1 + truncate_context_len + 1) + tokenizer.convert_tokens_to_ids(response) + [-1] * pad_length
                history_mask = [0] * (1 + truncate_context_len) + [1] * (1 + truncate_resp_len) + [0] * pad_length

                assert len(input_ids_NLG) == max_seq_length
                assert len(type_ids) == max_seq_length
                assert len(cls_mask_row) == max_seq_length
                assert len(cls_mask_col) == max_seq_length
                assert len(attention_masks_NLG) == max_seq_length
                assert len(attention_masks_NLU) == max_seq_length
                assert len(input_ids_NLU) == max_seq_length
                assert len(label_ids) == max_seq_length
                assert len(history_mask) == max_seq_length

                features.append(Input_feature(input_ids_NLG=input_ids_NLG,
                                              input_ids_NLU=input_ids_NLU,
                                              type_id=type_ids,
                                              input_masks_NLU=attention_masks_NLU,
                                              input_masks_NLG=attention_masks_NLG,
                                              cls_mask=[cls_mask_row, cls_mask_col],
                                              history_mask=history_mask,
                                              label_id=label_ids,
                                              curr_emotion_ids=context_emotion,
                                              fut_emotion_ids=response_emotion,
                                              intent_label=intent_label))

    torch.save(features, output_cache_file)


def build_decoding_data(cfg, tokenizer, input_json_file, output_cache_file, data_type,
                        speaker0_state=1, speaker1_state=2, speaker_state_pad=0, emotion_pad=0,
                        speaker0_token='<eos0>', speaker1_token='<eos1>', forward_truncate=False,
                        max_seq_length=128):
    features = []

    with open(input_json_file, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for data in tqdm(whole_data, desc=f'build {data_type} decoding feature'):
            utterence, token_type_ids = [], []
            label = data['label']

            for idx, sent in enumerate(data['utterence']):
                sent_token = tokenizer.tokenize(sent) + [speaker0_token if idx % 2 == 0 else speaker1_token]
                token_type_ids.append([speaker0_state if idx % 2 == 0 else speaker1_state] * len(sent_token))
                utterence.append(sent_token)

            # build feature
            for i in range(1, len(utterence)):
                if i % 2 == 0:
                    continue

                # build input for NLU
                input_ids_NLU = [tokenizer.cls_token_id]
                for no in range(i):
                    input_ids_NLU.extend(tokenizer.encode(data['utterence'][no]))
                input_ids_NLU = input_ids_NLU[:max_seq_length]

                # get context token and token type id
                context = [token for sent_token in utterence[:i] for token in sent_token]
                type_ids = [type_id for sent_type in token_type_ids[:i] for type_id in sent_type]

                # truncate to max_seq_length
                if forward_truncate:
                    context = [tokenizer.cls_token] + context[-(max_seq_length-2):] + [tokenizer.bos_token]
                    type_ids = [speaker_state_pad] + type_ids[-(max_seq_length-2):] + [speaker1_state]
                else:
                    context = [tokenizer.cls_token] + context[:max_seq_length-2] + [tokenizer.bos_token]
                    type_ids = [speaker_state_pad] + type_ids[:max_seq_length-2] + [speaker1_state]

                # convert to token_id
                context_ids = tokenizer.convert_tokens_to_ids(context)

                # get cls_mask
                cls_mask_row = [1] * (len(context_ids)-1) + [0]
                cls_mask_col = [1] + [0] * (len(context_ids) - 1)

                assert len(context_ids) == len(type_ids)
                assert len(context_ids) == len(cls_mask_row)
                assert len(context_ids) == len(cls_mask_col)

                features.append({
                    'input_ids_NLG': context_ids,
                    'input_ids_NLU': input_ids_NLU,
                    'token_type_ids': type_ids,
                    'cls_mask': [cls_mask_row, cls_mask_col],
                    'label': label
                })

    print("test num: ", len(features))
    f_w = open(output_cache_file, 'w', encoding='utf-8')
    json.dump(features, f_w)


def build_ref_txt(test_json_file, tokenizer: Tokenizer, ref_file):
    with open(test_json_file, 'r', encoding='utf-8') as f, open(ref_file, 'w', encoding='utf-8') as fo:
        whole_data = json.load(f)
        for element in tqdm(whole_data, desc='get reference'):
            for idx, utte in enumerate(element['utterence']):
                if idx % 2  == 0:
                    continue
                id_list = tokenizer.encode(utte)
                sentence = tokenizer.decode(id_list)
                fo.writelines(sentence+'\n')


def main(args):
    get_pretrained_embedding_weights(os.path.join(args.data_dir, "parsed_emotion_Ekman_{}.json"), args.glove_path, args.cache_dir)
    
    vocab_file = os.path.join(args.cache_dir, "vocab.json")
    tokenizer = Tokenizer(vocab_file)

    # build_ref_txt(os.path.join(cfg.data_dir, f"parsed_emotion_Ekman_intent_test.json"),
    #               tokenizer,
    #               './data/ref_trs_token.txt')
    
    tag_to_id_emotion = {'angry': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
    tag_to_id_intent = {'questioning': 0, 'acknowleding': 1, 'consoling': 2, 'agreeing': 3, 'encouraging': 4, 'sympathizing': 5, 'suggesting': 6, 'wishing': 7, 'neutral': 8}
    
    for d_type in ['test', 'train', 'valid']:
        input_json_file = os.path.join(args.data_dir, f"parsed_emotion_Ekman_intent_{d_type}.json")
        output_cache_file = os.path.join(args.cache_dir, f"cache_Trs_{d_type}_seq{args.max_length}")

        build_training_data(args, tokenizer, input_json_file, output_cache_file, d_type, tag_to_id_emotion, tag_to_id_intent)

        if d_type == 'test':
            output_json_file = os.path.join(args.cache_dir, f"decoding_Trs_seq{args.max_length}.json")
            build_decoding_data(args, tokenizer, input_json_file, output_json_file, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--glove_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()
    main(args)