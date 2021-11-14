import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import SequentialSampler, RandomSampler, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse

from build_GoEmotion_input_format import Input_feature

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models.pytorch_transformers import AdamW, WarmupLinearSchedule
from models.pytorch_transformers import BertForSequenceClassification, BertConfig



def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)


def train(cfg, model: BertForSequenceClassification, train_dataset: TensorDataset, validation_dataset: TensorDataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.train_batch_size)
    val_sampler = SequentialSampler(validation_dataset)
    val_dataloader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=cfg.eval_batch_size)

    steps_per_batch = len(train_dataloader)
    t_total = steps_per_batch * cfg.num_train_epochs
    warmup_steps = int(cfg.warmup_proportion * t_total)

    print("***** Running training *****")
    print(f"num eposhs = {cfg.num_train_epochs}")
    print(f"whole training steps = {t_total}")
    print(f"warmup steps = {warmup_steps}")
    print(f"train batch size = {cfg.train_batch_size}")
    print(f"learning rate = {cfg.lr}")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr, eps=cfg.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    record_txt = open(os.path.join(cfg.save_dir, "record.log"), 'w', encoding='utf-8')
    
    best_loss = np.Inf

    model.zero_grad()
    for epo in range(1, cfg.num_train_epochs + 1):
        model.train()
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        avg_train_loss = 0
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            avg_train_loss += loss.item()

            tqdm_bar.desc = f"epoch: [{epo}/{cfg.num_train_epochs}], step: [{step + 1}/{steps_per_batch}] lr: {scheduler.get_last_lr()[0]:.5f}, " \
                f"step loss: {loss.item():.4f}, avg loss: {avg_train_loss / (step + 1):.4f}"

        dev_loss, dev_acc = validation(cfg, model, val_dataloader)
        print(f"in epoch {epo}, dev loss: {dev_loss:.4f}, dev acc: {dev_acc:.4f}")

        if dev_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"best.pt"))
            best_loss = dev_loss

        record_txt.writelines(
            f"epoch: {epo}\t\ttrain loss: {avg_train_loss:.4f}\t\tdev loss: {dev_loss:.4f}\t\tdev acc: {dev_acc:.4f}\n")


def validation(cfg, model: BertForSequenceClassification, val_dataloader: DataLoader):
    dev_loss, count = 0, 0
    pred_label, gold_label = None, None
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            dev_loss += loss.item()
            count += 1

            if pred_label is None:
                pred_label = logits.detach().cpu().numpy()
                gold_label = inputs['labels'].detach().cpu().numpy()
            else:
                pred_label = np.append(pred_label, logits.detach().cpu().numpy(), axis=0)
                gold_label = np.append(gold_label, inputs['labels'].detach().cpu().numpy(), axis=0)

    pred_label = np.argmax(pred_label, axis=1)
    acc = accuracy_score(y_pred=pred_label, y_true=gold_label)
    label_list = ['angry', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    print(classification_report(y_pred=pred_label, y_true=gold_label, digits=4, target_names=label_list))
    dev_loss = dev_loss / count
    return dev_loss, acc


def calc_recall(cfg, model: BertForSequenceClassification, test_dataloader: DataLoader, num=3):
    dev_loss, count = 0, 0
    pred_label, gold_label = None, None
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(cfg.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            dev_loss += loss.item()
            count += 1

            _, pred_topk = torch.topk(logits.detach().cpu(), k=num)
            pred_topk = pred_topk.numpy()
            gold_id = inputs['labels'].detach().cpu().numpy()

            if pred_label is None:
                pred_label = pred_topk
                gold_label = gold_id
            else:
                pred_label = np.append(pred_label, pred_topk, axis=0)
                gold_label = np.append(gold_label, gold_id, axis=0)

    hit_num = 0
    for pred, gold in zip(pred_label, gold_label):
        if gold in pred:
            hit_num += 1

    print(f"top {num} recall: {hit_num / gold_label.shape[0]:.4f}")


def load_cache_examples(cfg, data_type):
    file_path = os.path.join(cfg.cache_dir, f"cache_classifier_goEmotion_{data_type}_len{cfg.max_length}")
    print(f"load cache from {file_path}")
    features = torch.load(file_path)

    input_ids = torch.Tensor([f.input_id for f in features]).long()
    input_masks = torch.Tensor([f.input_mask for f in features]).long()
    labels = torch.Tensor([f.label for f in features]).long()

    dataset = TensorDataset(input_ids, input_masks, labels)
    return dataset


def main(args):
    set_seed(args)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else 'cpu')
    args.device = device

    # build model
    print("build model ...")
    args.label_num = 7
    config = BertConfig.from_pretrained(args.bert_path, num_labels=args.label_num)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, config=config)
    model.to(args.device)

    if args.do_train:
        train_dataset = load_cache_examples(args, 'train')
        dev_dataset = load_cache_examples(args, 'dev')
        print("begin training ...")
        train(args, model, train_dataset, dev_dataset)

    if args.do_eval:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"best.pt")))
        test_dataset = load_cache_examples(args, 'dev')
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

        # calc_recall(args, model, test_loader, num=2)
        eval_loss, eval_acc = validation(args, model, test_loader)
        print(f"eval loss: {eval_loss:.4f}, eval acc: {eval_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_train", action='store_true')

    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--max_length", type=int, default=32)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)
