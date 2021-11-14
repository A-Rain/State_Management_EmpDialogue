import os
import json
import torch
from tqdm import tqdm
from typing import List
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.pytorch_transformers import BertForSequenceClassification, BertConfig, BertTokenizer



def _idx_to_tag_goEmotion():
    return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise', 6: 'neutral'}


def build_correct_data_format(cfg, tokenizer: BertTokenizer, data_type: str,
                              cls_token='[CLS]', pad_token_id=0, mask_padding_with_zero=True):
    whole_feature_ids, whole_feature_masks = [], []
    max_sequence_length = cfg.max_length
    file_name = os.path.join(cfg.data_dir, f"parsed_raw_{data_type}.json")
    with open(file_name, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for data in tqdm(whole_data, desc=f'build {data_type} format'):
            feature_ids, feature_masks = [], []
            for utterence in data['utterence']:
                tokens = [cls_token] + tokenizer.tokenize(utterence)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                origin_length = len(token_ids)
                if origin_length > max_sequence_length:
                    token_ids = token_ids[:max_sequence_length]
                    token_masks = [1] * max_sequence_length
                else:
                    pad_length = max_sequence_length - origin_length
                    token_ids = token_ids + pad_length * [pad_token_id]
                    token_masks = [1] * origin_length + [0 if mask_padding_with_zero else 1] * pad_length

                feature_ids.append(token_ids)
                feature_masks.append(token_masks)

            whole_feature_ids.append(torch.LongTensor(feature_ids))
            whole_feature_masks.append(torch.LongTensor(feature_masks))

    return whole_feature_ids, whole_feature_masks


def extract_emotion(cfg, model: BertForSequenceClassification, feature_ids: List[torch.Tensor],
                       feature_masks: List[torch.Tensor]):
    coarse_grained_emotion = []
    model.eval()
    with torch.no_grad():
        for input_ids, input_masks in tqdm(zip(feature_ids, feature_masks), desc='extract emotion'):
            output = model.forward(input_ids=input_ids.to(cfg.device),
                                   attention_mask=input_masks.to(cfg.device))
            logits = output[0]  # [Bsz, 7]
            coarse_grained = torch.argmax(logits, dim=-1).tolist()  # [Bsz, 1]
            coarse_grained_emotion.append(coarse_grained)

    return coarse_grained_emotion



def build_emotion_dialog_data(cfg, correspond_emotions, data_type, label_map):
    file_name = os.path.join(cfg.data_dir, f"parsed_raw_{data_type}.json")
    output_file = os.path.join(cfg.data_dir, f"parsed_emotion_Ekman_{data_type}.json")

    result = []

    with open(file_name, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for emotion_per_conv, data in tqdm(zip(correspond_emotions, whole_data), desc=f'build emotion {data_type} data'):
            new_data = {'conv_id': data['conv_id'],
                        'prompt': data['prompt'],
                        'label': data['label'],
                        'utterence': data['utterence']}

            coarse_grained_emotion =[label_map[emotion] for emotion in emotion_per_conv]
            new_data.update({'emotions': coarse_grained_emotion})

            result.append(new_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f)

def main(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else 'cpu')
    args.device = device

    # build tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    config = BertConfig.from_pretrained(args.bert_path, num_labels=7)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, config=config)
    model.to(args.device)

    best_emotion_classifier = os.path.join(args.save_dir, "best.pt")
    model.load_state_dict(torch.load(best_emotion_classifier))

    id_to_label = _idx_to_tag_goEmotion()

    for d_type in ['valid', 'test', 'train']:
        feature_ids, feature_masks = build_correct_data_format(args, tokenizer, d_type)
        coarse_grained_emotion = extract_emotion(args, model, feature_ids, feature_masks)
        build_emotion_dialog_data(args, coarse_grained_emotion, d_type, id_to_label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--max_length", type=int, default=32)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)

    

