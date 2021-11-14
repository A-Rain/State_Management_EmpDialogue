import json
import os
from tqdm import tqdm
from utils import simple_process_text
import argparse


def parse_data(data_dir, output_dir, data_type):
    '''
    this function aims at convert csv file into a json file for better understanding.
    '''
    json_data, example = [], dict()
    f = open(os.path.join(data_dir, f"{data_type}.csv"), 'r', encoding='utf-8')
    _ = f.readline()
    for element in tqdm(f, desc=f'parse {data_type}'):
        element = element.strip().split(",")
        uid = element[0]
        prompt = simple_process_text(element[3])
        label = element[2]
        utterence = simple_process_text(element[5])

        if len(example) == 0:
            example.update({'conv_id': uid, 'prompt': prompt, 'label': label, 'utterence': [utterence]})
        else:
            if uid == example['conv_id']:
                example['utterence'].append(utterence)
            else:
                json_data.append(example.copy())
                example = {'conv_id': uid, 'prompt': prompt, 'label': label, 'utterence': [utterence]}

    output_json_file = os.path.join(output_dir, f"parsed_raw_{data_type}.json")
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)
    print("finish parsed, whole num: ", len(json_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default='raw')
    parser.add_argument("--output_dir", type=str, default='output')
    args = parser.parse_args()
    for d_type in ['train', 'valid', 'test']:
        parse_data(args.raw_dir, args.output_dir, d_type)
