import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.pytorch_transformers import BertForSequenceClassification, BertConfig, BertTokenizer


import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict
from nltk.tokenize import sent_tokenize


def build_correct_intent_classifier_input(sentence: str, tokenizer: BertTokenizer, max_seq_length: int, max_sent_num=3):
    sent_list = sent_tokenize(sentence)
    input_ids, input_masks = [], []
    for sent in sent_list[:max_sent_num]:
        sent_id = tokenizer.encode(sent)[:max_seq_length]
        pad_length = max_seq_length - len(sent_id)
        input_masks.append([1] * len(sent_id) + [0] * pad_length)
        input_ids.append(sent_id + [tokenizer.pad_token_id] * pad_length)
    return torch.LongTensor(input_ids), torch.LongTensor(input_masks)


def get_intent(cfg, input_ids: torch.Tensor, input_masks: torch.Tensor, model: BertForSequenceClassification):
    output = model.forward(input_ids=input_ids.to(cfg.device),
                            attention_mask=input_masks.to(cfg.device))
    logits = output[0]  # [Bsz, 9]
    intent = set(torch.argmax(logits, dim=-1).tolist())
    return intent


def build_data_with_intent(cfg, model: BertForSequenceClassification, tokenizer: BertTokenizer, json_file, new_json_file, data_type, idx_to_tag: Dict[int, str]):
    model.eval()

    with open(json_file, 'r', encoding='utf-8') as f, torch.no_grad(), open(new_json_file, 'w', encoding='utf-8') as fo:
        whole_data = json.load(f)
        whole_data_update = []
        for element in tqdm(whole_data, desc=f'build {data_type} data'):
            new_element = {'conv_id': element['conv_id'],
                           'prompt': element['prompt'],
                           'label': element['label'],
                           'utterence': element['utterence'],
                           'emotions': element['emotions']}
            
            whole_intent = []

            for utterence in element['utterence']:
                input_ids, input_masks = build_correct_intent_classifier_input(utterence, tokenizer, cfg.max_length, cfg.max_sent_num)
                intent = get_intent(cfg, input_ids, input_masks, model)
                whole_intent.append([idx_to_tag[i] for i in intent])
            
            new_element.update({'intents': whole_intent})
            whole_data_update.append(new_element)
        
        json.dump(whole_data_update, fo)


def main(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else 'cpu')
    args.device = device

    # build tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    config = BertConfig.from_pretrained(args.bert_path, num_labels=9)
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best.pt")))
    model.to(device)

    idx_to_tag = {0: 'questioning', 1: 'acknowleding', 2: 'consoling', 3: 'agreeing', 4: 'encouraging', 
                  5: 'sympathizing', 6: 'suggesting', 7: 'wishing', 8: 'neutral'}
    
    for d_type in ['test', 'valid', 'train']:
        input_file = os.path.join(args.data_dir, f"parsed_emotion_Ekman_{d_type}.json")
        output_file = os.path.join(args.data_dir, f"parsed_emotion_Ekman_intent_{d_type}.json")
        build_data_with_intent(args, model, tokenizer, input_file, output_file, d_type, idx_to_tag)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--max_sent_num", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=32)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)


