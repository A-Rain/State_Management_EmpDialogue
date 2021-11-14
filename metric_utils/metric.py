import json
import argparse
import os
import re
import os
import numpy as np
import itertools
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from typing import List


class Embedding(object):
    def __init__(self, glove_path):
        self.m = KeyedVectors.load(os.path.join(glove_path, 'glove.6B.300d.model.bin'), mmap='r')
        self.unk = self.m.vectors.mean(axis=0)

    @property
    def w2v(self):
        return np.concatenate((self.m.syn0, self.unk[None, :]), axis=0)

    def __getitem__(self, key):
        try:
            return self.m.vocab[key].index
        except KeyError:
            return len(self.m.syn0)

    def vec(self, key):
        try:
            vectors = self.m.vectors
        except AttributeError:
            vectors = self.m.syn0
        try:
            return vectors[self.m.vocab[key].index]
        except KeyError:
            return self.unk


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def eval_emb_metrics(hypotheses: List[List[str]], references: List[List[str]], embedding_array: Embedding):
    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []

    for idx, hyp in enumerate(hypotheses):
        embs = np.array([embedding_array.vec(word) for word in hyp])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, maxemb,
                minemb)))

        emb_hyps.append(embs)
        avg_emb_hyps.append(avg_emb)
        extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for idx, ref in enumerate(references):
        embs = np.array([embedding_array.vec(word) for word in ref])
        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        # avg_emb = np.mean(embs,axis=0)
        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = np.array(list(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, maxemb,
                minemb)))
        emb_refs.append(embs)
        avg_emb_refs.append(avg_emb)
        extreme_emb_refs.append(extreme_emb)

    avg_cos_similarity = np.array([cos_sim(hyp, ref) for hyp, ref in zip(avg_emb_hyps, avg_emb_refs)])
    avg_cos_similarity = avg_cos_similarity.mean()
    extreme_cos_similarity = np.array([cos_sim(hyp, ref) for hyp, ref in zip(extreme_emb_hyps, extreme_emb_refs)])
    extreme_cos_similarity = extreme_cos_similarity.mean()

    scores = []
    for emb_ref, emb_hyp in zip(emb_refs, emb_hyps):
        simi_matrix = cosine_similarity(emb_ref, emb_hyp)
        dir1 = simi_matrix.max(axis=0).mean()
        dir2 = simi_matrix.max(axis=1).mean()
        scores.append((dir1 + dir2) / 2)
    greedy_scores = np.mean(scores)

    print("EmbeddingAverageCosineSimilairty: {0:.6f}".format(avg_cos_similarity))
    print("EmbeddingExtremeCosineSimilairty: {0:.6f}".format(extreme_cos_similarity))
    print("GreedyMatchingScore: {0:.6f}".format(greedy_scores))

    record = {"EmbeddingAverageCosineSimilairty": np.float(avg_cos_similarity),
              "EmbeddingExtremeCosineSimilairty": np.float(extreme_cos_similarity),
              "GreedyMatchingScore": np.float(greedy_scores)}
    return record


def calc_diversity(hypotheses: List[List[str]]):
    unigram_list = list(itertools.chain(*hypotheses))
    total_num_unigram = len(unigram_list)
    unique_num_unigram = len(set(unigram_list))
    bigram_list = []
    for hyp in hypotheses:
        hyp_bigram_list = list(zip(hyp[:-1], hyp[1:]))
        bigram_list += hyp_bigram_list
    total_num_bigram = len(bigram_list)
    unique_num_bigram = len(set(bigram_list))
    dist_1 = unique_num_unigram / total_num_unigram
    dist_2 = unique_num_bigram / total_num_bigram

    return dist_1, dist_2


def eval_diversity_metrics(hypotheses: List[List[str]], references: List[List[str]]):
    h_dist_1, h_dist_2 = calc_diversity(hypotheses)
    r_dist_1, r_dist_2 = calc_diversity(references)
    print("Dist-1 : Predicted {0:.6f} True {1:.6f}".format(h_dist_1, r_dist_1))
    print("Dist-2 : Predicted {0:.6f} True {1:.6f}".format(h_dist_2, r_dist_2))
    record = {"dist-1": h_dist_1, "dist-2": h_dist_2}
    return record


def calc_bleu(hyp_file: str, ref_file: str):
    cmd = f'perl metric_utils/multi-bleu.perl {ref_file}<{hyp_file}'
    f = os.popen(cmd, 'r')
    bleu_output = f.read()
    bleu_score = float(re.search(r"BLEU = (.+?),", bleu_output).group(1))
    print(f'BLEU: {bleu_score}')
    record = {'BLEU': bleu_score}
    return record

def calc_bleu_nltk(hypotheses: List[List[str]], references: List[List[List[str]]]):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1., 0., 0., 0.))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(.5, .5, 0., 0.))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(.333, .333, .333, 0.))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(.25, .25, .25, .25))
    print(f"Bleu_1: {bleu_1:.6f}")
    print(f"Bleu_2: {bleu_2:.6f}")
    print(f"Bleu_3: {bleu_3:.6f}")
    print(f"Bleu_4: {bleu_4:.6f}")
    record = {"Bleu_1": bleu_1, "Bleu_2": bleu_2, "Bleu_3": bleu_3, "Bleu_4": bleu_4}
    return record


def tokenize_raw_text(text_file, lower=True, space_token=False):
    tokenize_text = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            if space_token:
                if lower:
                    tokenize_text.append(list(map(str.lower, line.strip().split(" "))))
                else:
                    tokenize_text.append(line.strip().split(" "))
            else:
                if lower:
                    tokenize_text.append(list(map(str.lower, word_tokenize(line.strip()))))
                else:
                    tokenize_text.append(word_tokenize(line.strip()))
    return tokenize_text


def calc_BERTscore(hyp_file, ref_file, model_path, num_layer, rescale: bool, rescale_baseline_file: str):
    cmd = f'bert-score -r {ref_file} -c {hyp_file} --lang en --model {model_path} --num_layers {num_layer}'
    if rescale:
        cmd += f' --rescale_with_baseline --baseline_path {rescale_baseline_file}'
    f = os.popen(cmd, 'r')
    BERTscore_output = f.read()
    print(BERTscore_output)
    p_BERT = float(re.search(r"P: (.+?) R:", BERTscore_output).group(1))
    r_BERT = float(re.search(r"R: (.+?) F1:", BERTscore_output).group(1))
    f1_BERT = float(re.search(r"F1: (.+?)#", BERTscore_output.strip()+'#').group(1))
    print(f"P_BERT: {p_BERT}\nR_BERT: {r_BERT}\nF1_BERT: {f1_BERT}")
    record = {'P_BERT': p_BERT, 'R_BERT': r_BERT, 'F1_BERT': f1_BERT}
    return record


def compute_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower', action='store_true', help='lowercase or uppercase')
    parser.add_argument('--hyp', type=str, required=True, help='the generated responses file')
    parser.add_argument('--ref', type=str, required=True, help='the gold responses file')
    parser.add_argument('--output', type=str, required=True, help='output json result file')
    parser.add_argument('--space_token', action='store_true', help='tokenize by space')
    parser.add_argument('--emb_sim', action='store_true', help='calc word embedding similarity')
    parser.add_argument('--glove_path', type=str, default='./glove/', help='glove embedding for calc embedding similarity')
    parser.add_argument('--BERT_path', type=str, default='./roberta-large', help='roberta-large for calc BERTscore')
    parser.add_argument('--num_layers', type=int, default=17, help='using 17 layers of roberta-large for calc BERTscore')
    parser.add_argument('--rescale', action='store_true', help='whether rescale for calc BERTscore')
    args = parser.parse_args()

    hyps_origin = tokenize_raw_text(args.hyp, lower=args.lower, space_token=args.space_token)
    refs_origin = tokenize_raw_text(args.ref, lower=args.lower, space_token=args.space_token)
    assert len(hyps_origin) == len(refs_origin)

    # filter empty tokenized sentence
    hyps, refs = [], []
    avg_hyp_length, avg_ref_length = 0, 0
    for hyp, ref in zip(hyps_origin, refs_origin):
        if len(hyp) == 0:
            continue
        avg_hyp_length += len(hyp)
        avg_ref_length += len(ref)
        hyps.append(hyp)
        refs.append(ref)
    avg_hyp_length = avg_hyp_length / len(hyps)
    avg_ref_length = avg_ref_length / len(hyps)
    print(f"average hyp length: {avg_hyp_length}")
    print(f"average ref length: {avg_ref_length}")
    metric_result = {'average hyp length': avg_hyp_length, 'average ref length': avg_ref_length}

    # calc embedding similarity
    if args.emb_sim:
        glove_bin = os.path.join(args.glove_path, 'glove.6B.300d.model.bin')
        if os.path.exists(glove_bin):
            embd = Embedding(glove_path=args.glove_path)
            embed_metric = eval_emb_metrics(hyps, refs, embd)
            metric_result.update(embed_metric)
        else:
            print(f'{glove_bin} does not exists')
    
    # calc diverisy
    dist_metric = eval_diversity_metrics(hyps, refs)
    metric_result.update(dist_metric)
    
    # calc BLEU
    bleu_metric = calc_bleu(args.hyp, args.ref)
    metric_result.update(bleu_metric)

    # calc nltk bleu
    refs = [[ref] for ref in refs]
    bleu_nltk = calc_bleu_nltk(hyps, refs)
    metric_result.update(bleu_nltk)

    # calc BERTscore
    BERT_bin =  os.path.join(args.BERT_path, 'pytorch_model.bin')
    if not os.path.exists(BERT_bin):
        print(f"{BERT_bin} does not exists")
    else:
        rescale_baseline_file = os.path.join(args.BERT_path, "roberta-large.tsv")
        BERTscore_metric = calc_BERTscore(args.hyp, args.ref, args.BERT_path, args.num_layers, args.rescale, rescale_baseline_file)
        metric_result.update(BERTscore_metric)

    # save output
    with open(args.output, 'w', encoding='utf-8') as f_w:
        json.dump(metric_result, f_w)

if __name__ == "__main__":
    # print(os)
    compute_metrics()