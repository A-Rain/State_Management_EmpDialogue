import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import argparse
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from typing import List, Dict
from sklearn.metrics import classification_report, hamming_loss, f1_score, precision_score

from models.modeling_Emo_Intent_Trans_Trs import NLG_Trs_based
from models.modeling_emotion_intent_classifier_Trs import NLU_Trs_based
from models.pytorch_transformers import WarmupLinearSchedule, AdamW
from transformer_config import NLG_config, NLU_config
from preprocess import Input_feature, Tokenizer


def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)


def load_cache_examples(cfg, data_type):
    print(f"***** load {data_type} data cache *****")
    file_path = os.path.join(cfg.cache_dir, f"cache_Trs_{data_type}_seq{cfg.max_seq_length}")
    features = torch.load(file_path)

    input_ids_NLU = torch.Tensor([f.input_ids_NLU for f in features]).long()
    input_masks_NLU = torch.Tensor([f.input_masks_NLU for f in features]).float()
    curr_emotion_ids = torch.Tensor([[f.curr_emotion_ids] for f in features]).long()
    fut_emotion_ids = torch.Tensor([[f.fut_emotion_ids] for f in features]).long()
    intent_ids = torch.Tensor([f.intent_label for f in features]).float()

    input_ids_NLG = torch.Tensor([f.input_id_NLG for f in features]).long()
    input_masks_NLG = torch.Tensor([f.input_mask_NLG for f in features]).long()
    type_ids_NLG = torch.Tensor([f.type_id for f in features]).long()
    cls_masks = torch.Tensor([f.cls_mask for f in features]).long()
    history_mask = torch.Tensor([f.history_mask for f in features]).long()
    labels_ids = torch.Tensor([f.label_id for f in features]).long()
    
    dataset_NLG = TensorDataset(input_ids_NLU, input_masks_NLU, curr_emotion_ids, fut_emotion_ids, intent_ids,
                            input_ids_NLG, input_masks_NLG, type_ids_NLG, cls_masks, history_mask, labels_ids)
    
    dataset_NLU = TensorDataset(input_ids_NLU, input_masks_NLU, curr_emotion_ids, fut_emotion_ids, intent_ids)

    if data_type == 'train':
        sampler_NLG, sampler_NLU = RandomSampler(dataset_NLG), RandomSampler(dataset_NLU)
    else:
        sampler_NLG, sampler_NLU = SequentialSampler(dataset_NLG), SequentialSampler(dataset_NLU)
    dataloader_NLG = DataLoader(dataset_NLG, sampler=sampler_NLG, batch_size=cfg.train_batch_size_NLG if data_type == 'train' else cfg.eval_batch_size, 
                                drop_last=True if data_type != 'test' else False)
    dataloader_NLU = DataLoader(dataset_NLU, sampler=sampler_NLU, batch_size=cfg.train_batch_size_NLU if data_type == 'train' else cfg.eval_batch_size)
    
    return dataloader_NLG, dataloader_NLU


def schedule_sampling(cfg, curr_step, schedule_type='sig'):
    if schedule_type == 'lin':
        return max(cfg.sche_lin_eps, 1 - cfg.sche_lin_k * curr_step)
    elif schedule_type == 'exp':
        return np.power(cfg.sche_exp_k, curr_step)
    else:
        return cfg.sche_sig_k / (cfg.sche_sig_k + np.exp(curr_step / cfg.sche_sig_k))


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def calc_loss_NLU(cfg, current_logits: torch.Tensor, future_logits: torch.Tensor, intent_logits: torch.Tensor, current_gold_label: torch.Tensor, future_gold_logits: torch.Tensor, intent_gold_label: torch.Tensor):
    loss_fc = torch.nn.CrossEntropyLoss()
    loss_MultiLabel = torch.nn.BCELoss()

    loss_curr = loss_fc(current_logits, current_gold_label.view(-1))

    loss_fut = loss_fc(future_logits, future_gold_logits.view(-1))

    loss_intent = loss_MultiLabel(F.sigmoid(intent_logits), intent_gold_label)

    return (1-cfg.alpha) * loss_curr + cfg.alpha * loss_fut + cfg.beta * loss_intent, loss_curr, loss_fut, loss_intent


def calc_loss_NLG(cfg, lm_logits: torch.Tensor, lm_hard_label: torch.Tensor):
    loss_fc = torch.nn.CrossEntropyLoss(ignore_index=-1)
    shift_lm_logits = lm_logits[..., :-1, :].contiguous()
    shift_lm_logits = shift_lm_logits.view(-1, shift_lm_logits.size(-1))
    shift_lm_labels = lm_hard_label[..., 1:].contiguous()
    nll = loss_fc(shift_lm_logits, shift_lm_labels.view(-1))
    return nll


def pretrain_classifier(cfg, model: NLU_Trs_based, train_loader: DataLoader, valid_loader: DataLoader):
    steps_per_batch = len(train_loader)
    t_total = steps_per_batch * cfg.pretrain_NLU_epochs
    optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.pretrain_NLU_lr, eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

    print("pretraining NLU")
    model.zero_grad()
    for epo in range(1, cfg.pretrain_NLU_epochs + 1):
        model.train()  
        avg_train_loss, step = 0, 0
        tqdm_bar = tqdm(train_loader, desc="Training NLG")
        for batch in tqdm_bar:
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'gold_current_emotion_id': batch[2],
                      'forcing': True}
            curr_logits, fut_logits, intent_logits = model(**inputs)
            loss, _, _, _ = calc_loss_NLU(cfg, curr_logits, fut_logits, intent_logits, batch[2], batch[3], batch[-1])          

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            avg_train_loss += loss.item()
            step += 1

            tqdm_bar.desc = f"pretrain_cls|epoch:[{epo}/{cfg.pretrain_NLU_epochs}],step:[{step}/{steps_per_batch}],avg_loss:{avg_train_loss / step:.4f}"
    _ = validation_NLU(cfg, model, valid_loader)
    return model


def pretrain_lm(cfg, model: NLG_Trs_based, train_loader: DataLoader, dev_loader: DataLoader):
    steps_per_batch = len(train_loader)
    t_total = steps_per_batch * cfg.pretrain_NLG_epochs
    optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.pretrain_NLG_lr, eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)

    print("pretraining NLG")
    model.zero_grad()
    for epo in range(1, cfg.pretrain_NLG_epochs + 1):
        model.train()
        tqdm_bar = tqdm(train_loader, desc="Training NLG")
        avg_train_lm, step = 0, 0
        for batch in tqdm_bar:
            input_for_NLG = {'input_ids': batch[5].to(cfg.device),
                             'attention_mask': batch[6].to(cfg.device),
                             'token_type_ids': batch[7].to(cfg.device),
                             'mode': 3}
            logits = model(**input_for_NLG)
            loss = calc_loss_NLG(cfg, logits, batch[-1].to(cfg.device))

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            avg_train_lm += loss.item()
            step += 1
            tqdm_bar.desc = f"pretrain_lm|epoch:[{epo}/{cfg.pretrain_NLG_epochs}],step:[{step}/{steps_per_batch}],avg_loss:{avg_train_lm / step:.4f}"
        dev_loss = validation_backbone(cfg, model, dev_loader)
        print(f"pretrain_lm|epoch{epo}, dev_loss: {dev_loss:.4f}")
    return model


def validation_backbone(cfg, model_NLG: NLG_Trs_based, val_dataloader: DataLoader):
    dev_loss, count = 0, 0
    model_NLG.eval()
    with torch.no_grad():
        tqdm_bar = tqdm(val_dataloader, desc="Training NLG")
        for batch in tqdm_bar:
            input_for_NLG = {'input_ids': batch[5].to(cfg.device),
                             'attention_mask': batch[6].to(cfg.device),
                             'token_type_ids': batch[7].to(cfg.device),
                             'mode': 3}
            logits = model_NLG(**input_for_NLG)
            loss = calc_loss_NLG(cfg, logits, batch[-1].to(cfg.device))
            dev_loss += loss.item()
            count += 1
    return dev_loss / count


def train(cfg, model_NLU: NLU_Trs_based, model_NLG: NLG_Trs_based, 
          train_loader_NLG: DataLoader, valid_loader_NLG: DataLoader,
          train_loader_NLU: DataLoader, valid_loader_NLU: DataLoader):
    steps_per_batch_NLG = len(train_loader_NLG)
    t_total_NLG = steps_per_batch_NLG * cfg.alternate_num_epochs
    warmup_steps_NLG = int(cfg.warmup_proportion * t_total_NLG)
    optimizer_grouped_parameters_NLG = model_NLG.parameters()
    optimizer_NLG = AdamW(optimizer_grouped_parameters_NLG, lr=cfg.lr_NLG, eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay)
    scheduler_NLG = WarmupLinearSchedule(optimizer_NLG, warmup_steps=warmup_steps_NLG, t_total=t_total_NLG)

    steps_per_batch_NLU = len(train_loader_NLU)
    t_total_NLU = steps_per_batch_NLU * cfg.alternate_num_epochs
    warmup_steps_NLU = int(cfg.warmup_proportion * t_total_NLU)
    optimizer_grouped_parameters_NLU = model_NLU.parameters()
    optimizer_NLU = AdamW(optimizer_grouped_parameters_NLU, lr=cfg.lr_NLU, eps=cfg.adam_epsilon, weight_decay=cfg.weight_decay)
    scheduler_NLU = WarmupLinearSchedule(optimizer_NLU, warmup_steps=warmup_steps_NLU, t_total=t_total_NLU)

    record_txt = open(os.path.join(cfg.save_dir, "record.log"), 'w', encoding='utf-8')
    sigma_record = []

    print("***** Running training *****")
    print(f"GPU_id = {cfg.gpu_id}")
    print(f"schedule_sample = {cfg.schedule_type}")
    print(f"seed = {cfg.seed}")
    print(f"num epochs = {cfg.alternate_num_epochs}")
    print(f"whole training steps NLG = {t_total_NLG}")
    print(f"warmup steps NLG = {warmup_steps_NLG}")
    print(f"train batch size NLG = {cfg.train_batch_size_NLG}")
    print(f"learning rate NLG = {cfg.lr_NLG}")
    print(f"whole training steps NLU = {t_total_NLU}")
    print(f"warmup steps NLU = {warmup_steps_NLU}")
    print(f"train batch size NLU = {cfg.train_batch_size_NLU}")
    print(f"learning rate NLU = {cfg.lr_NLU}")
    print(f"alpha NLU = {cfg.alpha}")
    print(f"beta NLU = {cfg.beta}")

    best_loss = np.Inf

    global_step_NLG, global_step_NLU = 0, 0
    for epo in range(1, cfg.alternate_num_epochs+1):
        # train NLU first
        model_NLU.train()  
        avg_train_loss, avg_curr, avg_fut, avg_intent, step = 0, 0, 0, 0, 0
        tqdm_bar = tqdm(train_loader_NLU, desc="Train NLU")
        for batch in tqdm_bar:
            batch = tuple(t.to(cfg.device) for t in batch)
            p_use_gold = schedule_sampling(cfg, global_step_NLU, cfg.schedule_type)
            p_sample = np.random.rand()
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'gold_current_emotion_id': batch[2],
                      'forcing': True if p_use_gold > p_sample else False}
            curr_logits, fut_logits, intent_logits = model_NLU(**inputs)
            loss, loss_curr, loss_fut, loss_intent = calc_loss_NLU(cfg, curr_logits, fut_logits, intent_logits, batch[2], batch[3], batch[-1])          

            loss.backward()
            optimizer_NLU.step()
            scheduler_NLU.step()  # Update learning rate schedule
            model_NLU.zero_grad()

            avg_train_loss += loss.item()
            avg_curr += loss_curr.item()
            avg_fut += loss_fut.item()
            avg_intent += loss_intent.item()
            step += 1
            global_step_NLU += 1

            tqdm_bar.desc = f"epoch:[{epo}/{cfg.alternate_num_epochs}],step:[{step}/{steps_per_batch_NLU}]lr:{scheduler_NLU.get_last_lr()[0]:.5f}," \
                f"avg loss:{avg_train_loss / step:.4f},curr:{avg_curr / step:.4f},loss_fut:{avg_fut / step:.4f},loss_intent:{avg_intent / step:.4f}"
        dev_loss_NLU = validation_NLU(cfg, model_NLU, valid_loader_NLU, is_eval=False)

        # train NLG next
        model_NLU.eval()
        model_NLG.train()
        avg_train_loss, step = 0, 0
        tqdm_bar = tqdm(train_loader_NLG, desc="Train NLG")
        for batch in tqdm_bar:
            batch = tuple(t.to(cfg.device) for t in batch)
            p_use_gold = schedule_sampling(cfg, global_step_NLG, cfg.schedule_type)
            p_sample = np.random.rand()
            input_for_NLU = {'input_ids': batch[0],
                             'attention_mask': batch[1],
                             'gold_current_emotion_id': batch[2],
                             'gold_future_emotion_id': batch[3],
                             'gold_intent_id': batch[4],
                             'forcing': True if p_use_gold > p_sample else False}
            cur_emo_latent, fut_emo_latent, intent_latent = model_NLU.get_emotion_intent_latent(**input_for_NLU)

            input_for_NLG = {'input_ids': batch[5],
                             'attention_mask': batch[6],
                             'token_type_ids': batch[7],
                             'cls_mask': batch[8],
                             'history_mask': batch[9],
                             'current_emotion_latent': cur_emo_latent,
                             'future_emotion_latent': fut_emo_latent,
                             'intent_latent': intent_latent}
            logits, _ = model_NLG(**input_for_NLG)
            loss = calc_loss_NLG(cfg, logits, batch[-1])


            loss.backward()
            optimizer_NLG.step()
            scheduler_NLG.step()  # Update learning rate schedule
            model_NLG.zero_grad()

            avg_train_loss += loss.item()
            step += 1
            global_step_NLG += 1

            tqdm_bar.desc = f"epoch:[{epo}/{cfg.alternate_num_epochs}],step:[{step}/{steps_per_batch_NLG}]lr:{scheduler_NLU.get_last_lr()[0]:.5f},avg loss:{avg_train_loss / step:.4f}"
        dev_loss_NLG = validation_NLG(cfg, model_NLG, model_NLU, valid_loader_NLG)
        print(f"dev_loss_NLU: {dev_loss_NLU:.4f}, dev_loss_NLG: {dev_loss_NLG:.4f}")
        record_txt.writelines(f"dev_loss_NLU: {dev_loss_NLU:.4f}\t\tdev_loss_NLG: {dev_loss_NLG:.4f}\n")

        if dev_loss_NLG < best_loss:
            best_loss = dev_loss_NLG
            torch.save(model_NLU.state_dict(), os.path.join(cfg.save_dir, f"NLU_best.pt"))
            torch.save(model_NLG.state_dict(), os.path.join(cfg.save_dir, f"NLG_best.pt"))


def validation_NLU(cfg, model: NLU_Trs_based, val_dataloader: DataLoader, is_eval=True):
    dev_loss, count = 0, 0
    y_true_curr, y_pred_curr = [], []
    y_true_future, y_pred_future = [], []
    y_true_intent, y_pred_intent = [], []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            y_true_curr.extend((batch[2]).squeeze(1).tolist())
            y_true_future.extend((batch[3]).squeeze(1).tolist())
            y_true_intent.extend(batch[-1].tolist())

            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            curr_logits, fut_logits, intent_logits = model(**inputs)
            loss, _, _, _ = calc_loss_NLU(cfg, curr_logits, fut_logits, intent_logits, batch[2], batch[3], batch[-1])
            dev_loss += loss.item()
            count += 1

            y_pred_curr.extend(torch.argmax(curr_logits.cpu(), dim=-1).tolist())
            y_pred_future.extend(torch.argmax(fut_logits.cpu(), dim=-1).tolist())
            itt_pred_part1 = (F.sigmoid(intent_logits.cpu()) > 0.5).long()
            itt_pred_part2 = torch.topk(intent_logits.cpu(), k=1, dim=-1)[1]
            itt_pred_part2 = F.one_hot(itt_pred_part2, num_classes=9).squeeze(1)
            intent_pred = ((itt_pred_part1 + itt_pred_part2) > 0).long().tolist()
            y_pred_intent.extend(intent_pred)

    label_list = ['angry', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    curr_report = classification_report(y_true=y_true_curr, y_pred=y_pred_curr, digits=4, target_names=label_list, output_dict=True)
    future_report = classification_report(y_true=y_true_future, y_pred=y_pred_future, digits=4, target_names=label_list, output_dict=True)
    dev_loss = dev_loss / count

    ham_loss = hamming_loss(y_true_intent, y_pred_intent)
    mAP = precision_score(y_true_intent, y_pred_intent, average='micro')
    micro_f1 = f1_score(y_true_intent, y_pred_intent, average='micro')
    
    f1_curr, f1_fut = curr_report['weighted avg']['f1-score'], future_report['weighted avg']['f1-score']
    acc_curr, acc_fut = curr_report['accuracy'], future_report['accuracy']
    print(f"acc_curr: {acc_curr:.4f} | acc_fut: {acc_fut:.4f} | f1_curr: {f1_curr:.4f} | f1_fut: {f1_fut:.4f}")
    print(f"ham_loss: {ham_loss:.4f} | micro_f1: {micro_f1:.4f} | mAP: {mAP:.4f}")

    if is_eval:
        final_result = {
            'current': curr_report,
            'future': future_report,
            'intent': {'hamming': ham_loss, 'micro_f1': micro_f1, 'mAP': mAP},
            'loss': dev_loss
        }
        return final_result
    else:
        return dev_loss
        

def validation_NLG(cfg, model_NLG: NLG_Trs_based, model_NLU: NLU_Trs_based, val_dataloader: DataLoader):
    model_NLG.eval()
    model_NLU.eval()
    dev_loss, count = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='validation'):
            batch = tuple(t.to(cfg.device) for t in batch)
            input_for_NLU = {'input_ids': batch[0],
                             'attention_mask': batch[1]}
            cur_emo_latent, fut_emo_latent, intent_latent = model_NLU.get_emotion_intent_latent(**input_for_NLU)

            input_for_NLG = {'input_ids': batch[5],
                             'attention_mask': batch[6],
                             'token_type_ids': batch[7],
                             'cls_mask': batch[8],
                             'history_mask': batch[9],
                             'current_emotion_latent': cur_emo_latent,
                             'future_emotion_latent': fut_emo_latent,
                             'intent_latent': intent_latent}
            logits, _ = model_NLG(**input_for_NLG)
            loss = calc_loss_NLG(cfg, logits, batch[-1])
            dev_loss += loss.item()
            count += 1
    
    return dev_loss / count


def sample_sequence(cfg, model: NLG_Trs_based, tokenizer: Tokenizer, context_id: torch.Tensor, type_id: torch.Tensor, 
                    cls_mask: torch.Tensor, cur_emo_latent: torch.Tensor, fut_emo_latent: torch.Tensor, intent_latent: torch.Tensor, 
                    speaker1_state=2, decoding_strategy='sampling'):
    cls_mask_extra = torch.LongTensor([[[0], [0]]]).to(cfg.device)
    end_token_id = tokenizer.convert_tokens_to_ids('<eos1>')

    generated, past, sigma = context_id, None, None
    result = []
    for step in range(cfg.max_decode_length):
        input_for_NLG = {'input_ids': generated,
                         'token_type_ids': type_id,
                         'cls_mask': cls_mask,
                         'current_emotion_latent': cur_emo_latent,
                         'future_emotion_latent': fut_emo_latent,
                         'intent_latent': intent_latent,
                         'past': past,
                         'sigma': sigma}
        lm_logits, past, sigma = model.decoding(**input_for_NLG)
        next_token_logits = lm_logits[0, -1, :] / cfg.sampling_tau
        if decoding_strategy == 'sampling':
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=cfg.top_k, top_p=cfg.top_p)
            prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)
            next_token = next_token.unsqueeze(0)

        if next_token.item() == end_token_id and step >= cfg.min_decode_length:
            break

        result.append(next_token.item())
        generated = next_token.unsqueeze(0)
        type_id = torch.LongTensor([[speaker1_state]]).to(cfg.device)
        cls_mask = torch.cat((cls_mask, cls_mask_extra), dim=-1)

    text = tokenizer.decode(result, skip_special_tokens=True)
    text = text.replace("\n", "").replace("\r", "")
    # print(text)
    return text


def evaluation(cfg, tokenizer: Tokenizer, model_NLU: NLU_Trs_based, model_NLG: NLG_Trs_based, test_dataset: List[Dict], test_loader_NLU: DataLoader,
               test_loader_NLG: DataLoader, id_to_tag_emo: Dict[int, str], id_to_tag_itt: Dict[int, str]):
    suffix = 'greedy' if cfg.decoding_method == 'greedy' else f'sampling'
    hyp_file = os.path.join(cfg.save_dir, f"hyp_test_{suffix}.txt")
    check_file = os.path.join(cfg.save_dir, f"check_test_{suffix}.txt")
    result_file = os.path.join(cfg.save_dir, f"result_NLU.json")
    ref_txt = cfg.ref_file 

    model_NLU.eval()
    model_NLG.eval()
    with torch.no_grad(), open(check_file, 'w', encoding='utf-8') as f_check, open(hyp_file, 'w', encoding='utf-8') as f_hyp, open(ref_txt, 'r', encoding='utf-8') as f_r:
        for one_batch, ref in tqdm(zip(test_dataset, f_r), desc=f'begin decoding'):
            context_id = one_batch['input_ids_NLG']
            situation_label = one_batch['label']
            dialog_history = tokenizer.decode(context_id)

            token_type_id = torch.LongTensor(one_batch['token_type_ids']).unsqueeze(0).to(cfg.device)
            cls_mask = torch.LongTensor(one_batch['cls_mask']).unsqueeze(0).to(cfg.device)
            context_id = torch.LongTensor(context_id).unsqueeze(0).to(cfg.device)
            input_id_NLU = torch.LongTensor(one_batch['input_ids_NLU']).unsqueeze(0).to(cfg.device)

            cur_emo_latent, fut_emo_latent, intent_latent, pred_curr, pred_fut, pred_intent = \
                model_NLU.get_emotion_intent_latent(input_ids=input_id_NLU, eval_mode=True)
            
            pred_curr = id_to_tag_emo[pred_curr.item()]
            pred_fut = id_to_tag_emo[pred_fut.item()]
            pred_intent = [id_to_tag_itt[i.item()] for i in torch.where(pred_intent[0] == 1)[0]]
            pred_intent = " ".join(pred_intent)

            hyp = sample_sequence(cfg, model_NLG, tokenizer, context_id, token_type_id, cls_mask, cur_emo_latent, fut_emo_latent, intent_latent, 
                                  decoding_strategy=cfg.decoding_method)

            f_check.writelines(f"situation: {situation_label}\n")
            f_check.writelines(f"history: {dialog_history}\n")
            f_check.writelines(f"emotion_current: {pred_curr}, emotion_future: {pred_fut}\n")
            f_check.writelines(f"intent: {pred_intent}\n")
            f_check.writelines(f"hyp: {hyp.strip()}\n")
            f_check.writelines(f"ref: {ref.strip()}\n")
            f_check.writelines("\n")

            f_hyp.writelines(hyp.strip() + '\n')

    # compute regular metric

    result = validation_NLU(cfg, model_NLU, test_loader_NLU)  
    nll_loss = validation_NLG(cfg, model_NLG, model_NLU, test_loader_NLG)
    ppl = np.exp(nll_loss)    
    print(f"ppl: {ppl}")
    result.update({"ppl": ppl})

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f)



def main(args):
    set_seed(args)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else 'cpu')
    args.device = device

    # load pretrained embedding weights
    embed_weights = np.load(os.path.join(args.cache_dir, "embed_weights.npy"))
    print("weight size: ", embed_weights.shape)
    embed_weights = torch.from_numpy(embed_weights).float()

    print("load tokenizer ...")
    tokenizer = Tokenizer(vocab_file=os.path.join(args.cache_dir, "vocab.json"))

    print("load NLU_config ...")
    config_NLU = NLU_config()
    config_NLU.vocab_size = tokenizer.vocab_size
    config_NLU.n_layer = args.n_layer_NLU
    config_NLU.n_head = args.n_head_NLU

    print("load NLG_config ...")
    config_NLG = NLG_config()
    config_NLG.vocab_size = tokenizer.vocab_size
    config_NLG.n_layer = args.n_layer_NLG
    config_NLG.n_head = args.n_head_NLG

    # build NLU model
    print("build NLU model ...")
    print("load transition matrix ...")
    with open(os.path.join(args.data_dir, f'goEmotion_emotion_trans.mat'), 'r', encoding='utf-8') as f:
        emo_trans_mat = json.load(f)
        emo_trans_mat = torch.FloatTensor(emo_trans_mat).to(device)
    with open(os.path.join(args.data_dir, f'emotion_intent_trans.mat'), 'r', encoding='utf-8') as f:
        intent_trans_mat = json.load(f)
        intent_trans_mat = torch.FloatTensor(intent_trans_mat).to(device)
   
    model_NLU = NLU_Trs_based(config=config_NLU, 
                              pretrained_embed_weights=embed_weights,
                              emo_trans_matrix=emo_trans_mat, 
                              emo_intent_trans_matrix=intent_trans_mat, 
                              planning=args.planning)
    model_NLU.to(args.device)
    print("finish building NLU model ...")

    # build NLG model
    print("build NLG model ...")
    model_NLG = NLG_Trs_based(config=config_NLG, pretrained_embed_weights=embed_weights)
    model_NLG.to(args.device)
    print("finish building NLG model ...")

    if args.do_train:
        train_loader_NLG, train_loader_NLU = load_cache_examples(args, 'train')
        dev_loader_NLG, dev_loader_NLU = load_cache_examples(args, 'valid')

        # pretraining NLU and NLG first
        if args.pretrain_NLU_epochs != 0:
            pt_NLU_path = os.path.join(args.save_dir, "pt_NLU.pt")
            if not os.path.exists(pt_NLU_path):
                model_NLU = pretrain_classifier(args, model_NLU, train_loader_NLU, dev_loader_NLU)
                torch.save(model_NLU.state_dict(), pt_NLU_path)
            else:
                model_NLU.load_state_dict(torch.load(pt_NLU_path))
        
        if args.pretrain_NLG_epochs != 0:
            pt_NLG_path = os.path.join(args.save_dir, "pt_NLG.pt")
            if not os.path.exists(pt_NLG_path):
                model_NLG = pretrain_lm(args, model_NLG, train_loader_NLG, dev_loader_NLG)
                torch.save(model_NLG.state_dict(), pt_NLG_path)
            else:
                model_NLG.load_state_dict(torch.load(pt_NLG_path))
        
        train(args, model_NLU, model_NLG, train_loader_NLG, dev_loader_NLG, train_loader_NLU, dev_loader_NLU)
    
    if args.do_eval:
        idx_to_tag_itt = {0: 'questioning', 1: 'acknowleding', 2: 'consoling', 3: 'agreeing', 4: 'encouraging', 
                          5: 'sympathizing', 6: 'suggesting', 7: 'wishing', 8: 'neutral'}
        idx_to_tag_emo = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
    
        args.ref_file = os.path.join(args.data_dir, "ref_tokenize.txt")
        print("begin decoding ...")
        with open(os.path.join(args.cache_dir, f"decoding_Trs_seq{args.max_seq_length}.json")) as f:
            test_json = json.load(f)
        test_loader_NLG, test_loader_NLU = load_cache_examples(args, 'test')

        model_NLG.load_state_dict(torch.load(os.path.join(args.save_dir, f"NLG_best.pt")))
        model_NLU.load_state_dict(torch.load(os.path.join(args.save_dir, f"NLU_best.pt")))
        evaluation(args, tokenizer, model_NLU, model_NLG, test_json, test_loader_NLU, test_loader_NLG,
                   idx_to_tag_emo, idx_to_tag_itt)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)

    # several directory path
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)

    # train/eval
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_train", action='store_true')

    # whether use CUDA
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gpu_id", type=int, default=0)

    # max_seg_length
    parser.add_argument("--max_seq_length", type=int, default=128)

    # state manager setting
    parser.add_argument("--train_batch_size_NLU", type=int, default=16)
    parser.add_argument("--lr_NLU", type=float, default=7e-5)
    parser.add_argument("--planning", type=str, default="cat")
    parser.add_argument("--pretrain_NLU_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_NLU_epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--n_head_NLU", type=int, default=5)
    parser.add_argument("--n_layer_NLU", type=int, default=4)

    # response generator setting
    parser.add_argument("--train_batch_size_NLG", type=int, default=16)
    parser.add_argument("--lr_NLG", type=float, default=0.0015)
    parser.add_argument("--pretrain_NLG_lr", type=float, default=8e-5)
    parser.add_argument("--pretrain_NLG_epochs", type=int, default=3)
    parser.add_argument("--n_head_NLG", type=int, default=6)
    parser.add_argument("--n_layer_NLG", type=int, default=6)

    parser.add_argument("--eval_batch_size", type=int, default=16)

    # training epochs
    parser.add_argument("--alternate_num_epochs", type=int, default=15)

        # other optimizer setting
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # decoding setting
    parser.add_argument("--decoding_method", type=str, default="sampling")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sampling_tau", type=float, default=0.9)
    parser.add_argument("--min_decode_length", type=int, default=2)
    parser.add_argument("--max_decode_length", type=int, default=30)

    # schedule sampling
    parser.add_argument("--schedule_type", type=str, default="sig")
    parser.add_argument("--sche_lin_eps", type=float, default=0.1)
    parser.add_argument("--sche_exp_k", type=float, default=0.999)
    parser.add_argument("--sche_lin_k", type=float, default=0.001)
    parser.add_argument("--sche_sig_k", type=int, default=2500)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)





