import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_self_extended_mask(src_padding_mask):
    """
    :param src_padding_mask: [Bsz, seq_length]
    :return: extended_mask: [Bsz, seq_length, seq_length]
    """
    src_padding_mask = src_padding_mask.unsqueeze(2)
    return torch.bmm(src_padding_mask, src_padding_mask.transpose(1,2))


def get_cross_extended_mask(padding_mask1, padding_mask2):
    """
    :param padding_mask1: [Bsz, seq_len1]
    :param padding_mask2: [Bsz, seq_len2]
    :return: [Bsz, seq_len1, seq_len2]
    """
    return torch.bmm(padding_mask1.unsqueeze(2), padding_mask2.unsqueeze(1))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = 1- np.triu(np.ones(attn_shape), k=1)
    return torch.from_numpy(subsequent_mask)





class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, eps=1e-6):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.h = n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, d_ff, n_head, dropout, eps):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout, eps), 2)
        self.size = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class TransformerDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, d_ff, n_head, dropout, eps):
        super(TransformerDecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.src_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout, eps), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_model=512, d_ff=2048, n_head=8, n_layer=6, dropout=0.1, eps=1e-6):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(TransformerEncoderLayer(d_model, d_ff, n_head, dropout, eps), n_layer)
        self.norm = LayerNorm(d_model, eps=eps)


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        if mask is not None:
            extended_mask = get_self_extended_mask(mask)
        else:
            extended_mask = None

        for layer in self.layers:
            x = layer(x, extended_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_head=8, n_layer=6, dropout=0.1, eps=1e-6):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(TransformerDecoderLayer(d_model, d_ff, n_head, dropout, eps), n_layer)
        self.norm = LayerNorm(d_model, eps=eps)


    def forward(self, x, memory, src_mask, tgt_mask):
        if tgt_mask is not None:
            position_mask = subsequent_mask(tgt_mask.size(1)).type_as(tgt_mask)
            tgt_padding_mask = get_self_extended_mask(tgt_mask) * position_mask
        else:
            tgt_padding_mask = None

        if src_mask is not None and tgt_mask is not None:
            cross_padding_mask = get_cross_extended_mask(tgt_mask, src_mask)
        else:
            cross_padding_mask = None

        for layer in self.layers:
            x = layer(x, memory, cross_padding_mask, tgt_padding_mask)
        return self.norm(x)


class JointSentiTransformer(nn.Module):
    def __init__(self,
                 emotion_num,
                 type_num,
                 vocab_size,
                 leak_emotion_step,
                 d_model=512,
                 d_ff=2048,
                 n_head=8,
                 n_layer_enc=6,
                 n_layer_dec=6,
                 dropout=0.1,
                 max_position=1024,
                 temperature=1):
        super(JointSentiTransformer, self).__init__()
        self.leak_emotion_step = leak_emotion_step
        self.temperature = temperature

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_position, d_model)
        self.type_embedding = nn.Embedding(type_num, d_model, padding_idx=0)
        self.emotion_embedding = nn.Embedding(emotion_num, d_model, padding_idx=0)

        self.encoder = TransformerEncoder(d_model, d_ff, n_head, n_layer_enc, dropout)
        self.decoder = TransformerDecoder(d_model, d_ff, n_head, n_layer_dec, dropout)


    def resize_token_embeddings(self, new_num_tokens):
        old_num_tokens, old_embedding_dim = self.wte.weight.size()

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(self.wte.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.wte.weight.data[:num_tokens_to_copy, :]

        self.wte = new_embeddings


    def lm_head(self, hidden_states):
        """
        :param hidden_states: [Bsz, L, H], share the linear weights and word_embedding weights
        """
        return F.linear(hidden_states, self.wte.weight)

    def emotion_classify(self, cls_states):
        """
        :param cls_states: [Bsz, H], share the linear weights and emotion_embedding weights
        remember to get over pad
        """
        return F.linear(cls_states, self.emotion_embedding.weight[1:])

    @staticmethod
    def add_gumbel(o_t, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = o_t.new_zeros(o_t.size())
        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t


    def get_pred_emotion_embed_matrix(self, input_ids):
        """
        since dimension for embedding table is [emotion_size+1, embed_dim], dimension of input_ids should be [Bsz, emotion_size], here +1 means pad
        return: a embedding matrix: [Bsz, embed_dim]
        """
        return F.linear(input_ids, self.emotion_embedding.weight[1:].t(), bias=None)


    def _embed(self, input_ids, token_type_ids=None, emotion_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_embedding = self.wte(input_ids) + self.wpe(position_ids) + self.type_embedding(token_type_ids)
        if emotion_ids is not None:
            input_embedding = input_embedding + self.emotion_embedding(emotion_ids)

        return input_embedding


    def forward(self, encoder_inputs, decoder_inputs, step, gold_response_emotion_id=None, src_padding_mask=None, tgt_padding_mask=None):
        encoder_input = self._embed(**encoder_inputs)
        encoder_output = self.encoder(encoder_input, src_padding_mask)

        emotion_logits = self.emotion_classify(encoder_output[:, 0])

        decoder_input = self._embed(**decoder_inputs)
        # 1: means we do not consider [CLS] special tokens
        decoder_output = self.decoder(decoder_input, encoder_output[:, 1:], src_mask=src_padding_mask[:, 1:], tgt_mask=tgt_padding_mask)

        if gold_response_emotion_id is not None:
            resp_emotion_embed = self.emotion_embedding(gold_response_emotion_id)
            if step >= self.leak_emotion_step:
                gumble_logits = self.add_gumbel(emotion_logits)  # [Bsz, emotion_num]
                gumble_distri = torch.softmax(gumble_logits * self.temperature, dim=-1)
                emotion_pred = self.get_pred_emotion_embed_matrix(gumble_distri)  # [Bsz, emotion_embed]
                emotion_pred = emotion_pred.unsqueeze(1)  # [Bsz, 1, emotion_embed]
                # get one-zero matrix
                one_zero_matrix = (resp_emotion_embed != 0).float()
                resp_emotion_embed = emotion_pred * one_zero_matrix

            decoder_output = decoder_output + resp_emotion_embed

        lm_logits = self.lm_head(decoder_output)
        return emotion_logits, lm_logits


    def validate(self, encoder_inputs, decoder_inputs, gold_response_emotion_id=None, src_padding_mask=None, tgt_padding_mask=None):
        encoder_input = self._embed(**encoder_inputs)
        encoder_output = self.encoder(encoder_input, src_padding_mask)

        emotion_logits = self.emotion_classify(encoder_output[:, 0])

        decoder_input = self._embed(**decoder_inputs)
        decoder_output = self.decoder(decoder_input, encoder_output[:, 1:], src_mask=src_padding_mask[:, 1:], tgt_mask=tgt_padding_mask)

        if gold_response_emotion_id is not None:
            pred_emotion_id = torch.argmax(emotion_logits, dim=-1) + 1  # [Bsz], +1 means consider pad
            pred_emotion_embed = self.emotion_embedding(pred_emotion_id).unsqueeze(1)

            resp_emotion_embed = self.emotion_embedding(gold_response_emotion_id)
            one_zero_matrix = (resp_emotion_embed != 0).float()

            # add predict response emotion to transformer output
            resp_emotion_embed = pred_emotion_embed * one_zero_matrix
            decoder_output = decoder_output + resp_emotion_embed

        lm_logits = self.lm_head(decoder_output)
        return emotion_logits, lm_logits


    def decoding(self, encoder_cache, pred_response_emotion_ids, decoder_inputs, src_padding_mask):
        decoder_input = self._embed(**decoder_inputs)
        tgt_padding_mask = torch.ones(decoder_inputs['input_ids'].size()).type_as(src_padding_mask)
        decoder_output = self.decoder(decoder_input, encoder_cache[:, 1:], src_mask=src_padding_mask[:, 1:], tgt_mask=tgt_padding_mask)

        decoder_output = decoder_output + self.emotion_embedding(pred_response_emotion_ids)
        lm_logits = self.lm_head(decoder_output)
        return lm_logits


class VallinaTransformer(nn.Module):
    def __init__(self,
                 type_num,
                 vocab_size,
                 d_model=512,
                 d_ff=2048,
                 n_head=8,
                 n_layer_enc=6,
                 n_layer_dec=6,
                 dropout=0.1,
                 max_position=1024):
        super(VallinaTransformer, self).__init__()

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_position, d_model)
        self.type_embedding = nn.Embedding(type_num, d_model, padding_idx=0)

        self.encoder = TransformerEncoder(d_model, d_ff, n_head, n_layer_enc, dropout)
        self.decoder = TransformerDecoder(d_model, d_ff, n_head, n_layer_dec, dropout)

    def resize_token_embeddings(self, new_num_tokens):
        old_num_tokens, old_embedding_dim = self.wte.weight.size()

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(self.wte.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.wte.weight.data[:num_tokens_to_copy, :]

        self.wte = new_embeddings

    def lm_head(self, hidden_states):
        """
        :param hidden_states: [Bsz, L, H], share the linear weights and word_embedding weights
        """
        return F.linear(hidden_states, self.wte.weight)


    def _embed(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_embedding = self.wte(input_ids) + self.wpe(position_ids) + self.type_embedding(token_type_ids)

        return input_embedding

    def forward(self, encoder_inputs, decoder_inputs, src_padding_mask=None, tgt_padding_mask=None):
        encoder_input = self._embed(**encoder_inputs)
        encoder_output = self.encoder(encoder_input, src_padding_mask)

        decoder_input = self._embed(**decoder_inputs)
        # 1: means we do not consider [CLS] special tokens
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask=src_padding_mask, tgt_mask=tgt_padding_mask)

        lm_logits = self.lm_head(decoder_output)
        return lm_logits

    def validate(self, encoder_inputs, decoder_inputs, src_padding_mask=None, tgt_padding_mask=None):
        encoder_input = self._embed(**encoder_inputs)
        encoder_output = self.encoder(encoder_input, src_padding_mask)

        decoder_input = self._embed(**decoder_inputs)
        decoder_output = self.decoder(decoder_input, encoder_output, src_mask=src_padding_mask, tgt_mask=tgt_padding_mask)

        lm_logits = self.lm_head(decoder_output)
        return lm_logits

    def decoding(self, encoder_cache, decoder_inputs, src_padding_mask):
        decoder_input = self._embed(**decoder_inputs)
        tgt_padding_mask = torch.ones(decoder_inputs['input_ids'].size()).type_as(src_padding_mask)
        decoder_output = self.decoder(decoder_input, encoder_cache, src_mask=src_padding_mask, tgt_mask=tgt_padding_mask)

        lm_logits = self.lm_head(decoder_output)
        return lm_logits

    # if __name__ == '__main__':
#     model = JointSentiTransformer(emotion_num=2,
#                                   type_num=2,
#                                   vocab_size=5,
#                                   d_model=8,
#                                   n_head=1,
#                                   d_ff=16,
#                                   n_layer_dec=2,
#                                   n_layer_enc=2,
#                                   leak_emotion_step=4)
#
#     input_id_enc = torch.LongTensor([[2,3,1,0,0],[4,1,3,1,0]])
#     input_mask_enc = torch.LongTensor([[1,1,1,0,0],[1,1,1,1,0]])
#     input_id_dec = torch.LongTensor([[3,1,0,0],[4,2,1,0]])
#     input_mask_dec = torch.LongTensor([[1,1,0,0],[1,1,1,0]])
#     type_id = torch.LongTensor([[1,1,1,1,1],[1,1,1,1,1]])
#     type_id_dec = torch.LongTensor([[1,1,1,1],[1,1,1,1]])
#
#     enc_inputs = {
#         'input_ids': input_id_enc,
#         'token_type_ids': type_id,
#         'emotion_ids': None,
#         'position_ids': None
#     }
#
#     dec_inputs = {
#         'input_ids': input_id_dec,
#         'token_type_ids': type_id_dec,
#         'emotion_ids': None,
#         'position_ids': None
#     }
#     model.forward(enc_inputs, dec_inputs, step=1, src_padding_mask=input_mask_enc, tgt_padding_mask=input_mask_dec)
#


