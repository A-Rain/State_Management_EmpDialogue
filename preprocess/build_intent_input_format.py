import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import re
import torch
from nltk.tokenize import sent_tokenize
import argparse

import json
from tqdm import tqdm
from collections import Counter

from models.pytorch_transformers import BertTokenizer


class Input_intent_feature(object):
    def __init__(self, input_ids, attention_masks, label):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label = label


def statistics_on_sent_num(json_file, d_type):
    a = Counter()
    num_list = []
    with open(json_file, 'r', encoding='utf-8') as f:
        whole_data = json.load(f)
        for element in tqdm(whole_data, desc='statistics'):
            for utte in element['utterence']:
                sent_num = len(sent_tokenize(utte))
                num_list.append(sent_num)
    
    a.update(num_list)
    print(f"{d_type}:")
    print(a)


def statistics_on_overlap(data_dir, intent_dir):
    utte_pool = set()
    for d_type in ['train', 'test', 'valid']:
        json_file = os.path.join(data_dir, f"parsed_raw_{d_type}.json")
        with open(json_file, 'r', encoding='utf-8') as f:
            whole_data = json.load(f)
            for element in tqdm(whole_data, desc='statistics'):
                for utte in element['utterence']:
                    for sub_utte in sent_tokenize(utte):
                        utte_pool.add(sub_utte.strip().lower())
    
    intent_pool = set()
    for d_type in ['train', 'test', 'validation']:
        for intent in ['questioning', 'acknowleding', 'consoling', 'agreeing', 'encouraging', 'sympathizing', 'suggesting', 'wishing', 'neutral']:
            file_name = os.path.join(intent_dir, d_type, f"{intent}.txt")
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'read {d_type} {intent}'):
                    intent_sent = line.split("<SEP>")[1].strip().lower()
                    intent_pool.add(intent_sent)
    
    print(len(utte_pool & intent_pool) / len(utte_pool))

    
def build_training_data(cfg, tokenizer: BertTokenizer, d_type, output_cache_file, tag_to_id, max_seq_length=32):
    unique_set = set()
    features = []
    for intent in ['questioning', 'acknowleding', 'consoling', 'agreeing', 'encouraging', 'sympathizing', 'suggesting', 'wishing', 'neutral']:
        file_name = os.path.join(cfg.data_dir, d_type, f"{intent}.txt")
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'read {d_type} {intent}'):
                intent_sent = line.split("<SEP>")[1].strip()
                if intent_sent in unique_set:
                    continue

                unique_set.add(intent_sent)

                intent_sent_ids = [tokenizer.cls_token_id] + tokenizer.encode(intent_sent)
                intent_sent_ids = intent_sent_ids[:max_seq_length]
                pad_length = max_seq_length - len(intent_sent_ids)
                attention_masks = [1] * len(intent_sent_ids) + [0] * pad_length
                intent_sent_ids = intent_sent_ids + [tokenizer.pad_token_id] * pad_length
                features.append(Input_intent_feature(input_ids=intent_sent_ids,
                                                     attention_masks=attention_masks,
                                                     label=tag_to_id[intent]))

    print(f"num: {len(features)}")
    torch.save(features, output_cache_file)        

def main(args):
    tag_to_id = {'questioning': 0, 'acknowleding': 1, 'consoling': 2, 'agreeing': 3, 'encouraging': 4, 
                 'sympathizing': 5, 'suggesting': 6, 'wishing': 7, 'neutral': 8}
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    for d_type in ['train', 'validation', 'test']:
        output_cache_file = os.path.join(args.cache_dir, f"cache_intent_{d_type}_seq{args.max_length}")
        build_training_data(args, tokenizer, d_type, output_cache_file, tag_to_id, max_seq_length=args.max_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=32)
    args = parser.parse_args()
    main(args)
    