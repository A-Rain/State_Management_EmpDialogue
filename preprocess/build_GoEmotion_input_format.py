import json
import re
from tqdm import tqdm
import os
import torch
from collections import Counter
from typing import Dict
import argparse

from utils import simple_process_text

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.pytorch_transformers import BertTokenizer


class Input_feature(object):
    def __init__(self, input_id, input_mask, label):
        self.input_id = input_id
        self.input_mask = input_mask
        self.label = label


def Ekman_mapping(emotion_class_file, map_json_file):
    emo2ind = dict()
    with open(emotion_class_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            emo2ind.update({line.strip(): idx})

    Ekman_dict = dict()
    with open(map_json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for idx, (k, v) in enumerate(json_data.items()):
            print(f"{k}: {idx}")
            for ele in v:
                Ekman_dict.update({emo2ind[ele]: idx})

    return Ekman_dict


def build_GoEmotion_data(tokenizer: BertTokenizer, data_dir: str, output_dir: str, ekman_map: Dict[int, int], data_type: str, max_seq_length=64):
    features = []
    with open(os.path.join(data_dir, f"{data_type}.tsv"), 'r' , encoding='utf-8') as f:
        for line in tqdm(f, desc=f'build {data_type} feature'):
            line = line.split('\t')
            if len(line[1].split(',')) > 1:
                continue

            emotion_label = ekman_map[int(line[1])]
            text_token = [tokenizer.cls_token] + tokenizer.tokenize(simple_process_text(line[0]))
            text_id = tokenizer.convert_tokens_to_ids(text_token)

            if (len(text_id) < max_seq_length):
                input_masks = [1] * len(text_id) + [0] * (max_seq_length - len(text_id))
                text_id += [tokenizer.pad_token_id] * (max_seq_length - len(text_id))

            else:
                input_masks = [1] * max_seq_length
                text_id = text_id[:max_seq_length]


            assert len(text_id) == max_seq_length
            assert len(input_masks) == max_seq_length

            features.append(Input_feature(input_id=text_id,
                                          input_mask=input_masks,
                                          label=emotion_label))

    output_file = os.path.join(output_dir, f"cache_classifier_goEmotion_{data_type}_len{max_seq_length}")
    print(f"item number: {len(features)}")
    torch.save(features, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=32)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    new_label_map = Ekman_mapping(os.path.join(args.data_dir, 'emotions.txt'), os.path.join(args.data_dir, 'ekman_mapping.json'))
    for d_type in ['train', 'dev', 'test']:
        build_GoEmotion_data(tokenizer, args.data_dir, args.cache_dir, new_label_map, d_type, max_seq_length=args.max_length)