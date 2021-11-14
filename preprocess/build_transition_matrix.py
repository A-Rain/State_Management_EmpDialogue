import os
import json
from tqdm import tqdm
from typing import Dict
import torch
import argparse

def get_emotion_intent_transition_matrix(dialog_json_dir, tag_to_idx_emotion: Dict[str, int], tag_to_idx_intent: Dict[str, int], 
                                         emotion_num=7, intent_num=9):
    matrix = torch.zeros(emotion_num, intent_num)
    for d_type in ['train', 'valid']:
        dialog_json_file = os.path.join(dialog_json_dir, f"parsed_emotion_Ekman_intent_{d_type}.json")
        with open(dialog_json_file, 'r', encoding='utf-8') as f:
            whole_data = json.load(f)
            for data in tqdm(whole_data, desc='build matrix'):
                emotion, intents = data['emotions'], data['intents']
                for i in range(len(emotion)):
                    if i % 2 == 0:
                        continue

                    x_idx = tag_to_idx_emotion[emotion[i-1]]
                    for intent in intents[i]:
                        y_idx = tag_to_idx_intent[intent]
                        matrix[x_idx][y_idx] += 1
    
    # print(matrix.long())
    a = matrix / matrix.sum(dim=1).unsqueeze(1).expand(7,9)
    # print(a)
    return a


def get_emotion_emotion_transition_matrix(dialog_json_dir, tag_to_idx_emotion: Dict[str, int], tag_to_idx_intent: Dict[str, int], 
                                         emotion_num=7, intent_num=9):
    matrix = torch.zeros(emotion_num, emotion_num)
    for d_type in ['train', 'valid']:
        dialog_json_file = os.path.join(dialog_json_dir, f"parsed_emotion_Ekman_intent_{d_type}.json")
        with open(dialog_json_file, 'r', encoding='utf-8') as f:
            whole_data = json.load(f)
            for data in tqdm(whole_data, desc='build matrix'):
                emotion = data['emotions']
                for i in range(len(emotion)):
                    if i % 2 == 0:
                        continue

                    x_idx = tag_to_idx_emotion[emotion[i-1]]
                    y_idx = tag_to_idx_emotion[emotion[i]]
                    matrix[x_idx][y_idx] += 1
    
    # print(matrix.long())
    a = matrix / matrix.sum(dim=1).unsqueeze(1).expand(7,7)
    # print(a)
    return a


def main(args):
    tag_to_id_emotion = {'angry': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
    tag_to_id_intent = {'questioning': 0, 'acknowleding': 1, 'consoling': 2, 'agreeing': 3, 'encouraging': 4, 'sympathizing': 5, 'suggesting': 6, 'wishing': 7, 'neutral': 8}
    
    matrix_emo_itt = get_emotion_intent_transition_matrix(args.data_dir, tag_to_id_emotion, tag_to_id_intent)
    with open(os.path.join(args.data_dir, f'emotion_intent_trans.mat'), 'w', encoding='utf-8') as f:
        json.dump(matrix_emo_itt.tolist(), f)
    
    matrix_emo_emo = get_emotion_emotion_transition_matrix(args.data_dir, tag_to_id_emotion, tag_to_id_intent)
    with open(os.path.join(args.data_dir, f'goEmotion_emotion_trans.mat'), 'w', encoding='utf-8') as f:
        json.dump(matrix_emo_emo.tolist(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
    