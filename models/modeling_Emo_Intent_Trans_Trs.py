import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pytorch_transformers.modeling_utils import Conv1D 

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, causal_mask=None, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)  # (batch, head, seq_length, seq_length)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        causal_mask = causal_mask[:, :, ns - nd:ns, :ns]
        w = w * causal_mask - 1e4 * (1 - causal_mask)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, cls_mask=None, layer_past=None, attention_mask=None, head_mask=None):
        """
        cls_mask: [Bsz, seq_length], used for mask dialog history. this mask aims at make [CLS] only focus
        on response
        """
        if cls_mask is not None:
            Bsz, _, seq_len = cls_mask.shape
            # add cls_mask into causal mask (self.bias)
            causal_mask = self.bias[:, :, :seq_len, :seq_len].repeat(Bsz, 1, 1, 1)
            causal_mask[:, :, 0] = cls_mask[:, 0].unsqueeze(1)
            causal_mask[:, :, :, 0] = cls_mask[:, 1].unsqueeze(1)
        else:
            causal_mask = self.bias

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)  # (batch, head, seq_length, head_features)
        key = self.split_heads(key, k=True)  # (batch, head, head_features, seq_length)
        value = self.split_heads(value)  # (batch, head, seq_length, head_features)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, causal_mask, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class Block_NLG(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block_NLG, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, cls_mask=None, layer_past=None, attention_mask=None):
        output_attn = self.attn(self.ln_1(x),
                                cls_mask=cls_mask,
                                layer_past=layer_past,
                                attention_mask=attention_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class NLG_Backbone(nn.Module):
    def __init__(self, config):
        super(NLG_Backbone, self).__init__()
        self.nlayer = config.n_layer
        self.h = nn.ModuleList([Block_NLG(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_embed, cls_mask=None, past=None, attention_mask=None, decoding=False):
        if past is None:
            past = [None] * len(self.h)
        
        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = input_embed
        output_shape = input_embed.size()

        presents = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            outputs = block(hidden_states,
                            cls_mask=cls_mask,
                            layer_past=layer_past,
                            attention_mask=attention_mask)

            hidden_states, present = outputs[:2]
            presents = presents + (present,)
    
        # pay attention! we don't put ''presents`` into outputs while training because it will cause dimension mismatching
        if decoding:
            # in decoding mode
            outputs = (hidden_states, presents)
        else:
            # in parallel mode
            outputs = (hidden_states,)
        
        return outputs


class NLG_Trs_based(nn.Module):
    def __init__(self, config, pretrained_embed_weights):
        super(NLG_Trs_based, self).__init__()
        self.transformer = NLG_Backbone(config)
        self.wte = nn.Embedding.from_pretrained(pretrained_embed_weights, padding_idx=0, freeze=False)
        self.wpe = nn.Embedding(config.max_position, config.n_embd)
        self.token_type_embedding = nn.Embedding(config.type_num + 1, config.n_embd, padding_idx=0)
        self.dropout = nn.Dropout(config.embd_pdrop)

        self.hidden_size = config.n_embd
        self.gate_layer = nn.Sequential(nn.Linear(config.n_embd, 1), nn.Sigmoid())

        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size, 1, bias=False)

    def lm_head(self, hidden_states):
        """
        :param hidden_states: [Bsz, L, H], share the linear weights and word_embedding weights
        """
        return F.linear(hidden_states, self.wte.weight)

    def fuse_emotion(self, hidden_states, current_emotion_latent, future_emotion_latent, history_mask_extended=None):
        """
        hidden_states:          [Bsz, L, H]
        history_mask_extended:  [Bsz, L, H] or None
        current_emotion_latent: [Bsz, 1, H]
        future_emotion_latent:  [Bsz, 1, H]
        """
        if history_mask_extended is not None:
            current_emotion_latent = current_emotion_latent * history_mask_extended
            future_emotion_latent = future_emotion_latent * history_mask_extended
        
        hidden_states = self.w1(hidden_states + current_emotion_latent) + F.tanh(self.w2(hidden_states)) * future_emotion_latent
        # hidden_states = self.w1(torch.cat((hidden_states, current_emotion_latent), dim=-1))
        emo_fused_logits = self.lm_head(hidden_states)
        return emo_fused_logits

    def fuse_intent(self, hidden_states, intent_latent, history_mask_extended=None):
        """
        hidden_states:          [Bsz, L, H]
        history_mask_extended:  [Bsz, L, H] or None
        intent_latent:          [Bsz, 1, H]
        """
        if history_mask_extended is not None:
            intent_latent = intent_latent * history_mask_extended
        hidden_states = hidden_states * intent_latent + intent_latent
        intent_fused_logits = self.lm_head(hidden_states)
        return intent_fused_logits
    
    def fuse_all(self, cls_states, fused_emotion_logits, fused_intent_logits, sigma=None):
        if sigma is None:
            sigma = self.gate_layer(cls_states).unsqueeze(2)  # [Bsz, 1, 1]
        return fused_emotion_logits * sigma + fused_intent_logits * (1 - sigma), sigma
    
    def _embed(self, input_ids, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
        else:
            past_length = past[0][0].size(-2)
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.token_type_embedding(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.dropout(hidden_states)
        return hidden_states


    def forward(self, input_ids, history_mask=None, past=None, cls_mask=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, decoding=False, 
                current_emotion_latent=None, future_emotion_latent=None, intent_latent=None, mode=0):
        input_embedding = self._embed(input_ids, token_type_ids, past)
        transformer_outputs = self.transformer(input_embed=input_embedding,
                                               cls_mask=cls_mask,
                                               past=past,
                                               attention_mask=attention_mask,
                                               decoding=decoding)
        hidden_states = transformer_outputs[0]

        # 0 means fuse all, 1 means only fuse emotion, 2 means only fuse intent, 3 means no fuse
        if mode == 0:
            history_mask_extended = history_mask.unsqueeze(2).repeat(1,1,self.hidden_size)
            cls_states = hidden_states[:, 0]
            fused_emotion_logits = self.fuse_emotion(hidden_states, current_emotion_latent, future_emotion_latent, history_mask_extended)
            fused_intent_logits = self.fuse_intent(hidden_states, intent_latent, history_mask_extended)
            logits, sigma = self.fuse_all(cls_states, fused_emotion_logits, fused_intent_logits)
            return logits, sigma
        elif mode == 1:
            history_mask_extended = history_mask.unsqueeze(2).repeat(1,1,self.hidden_size)
            logits = self.fuse_emotion(hidden_states, current_emotion_latent, future_emotion_latent, history_mask_extended)
            return logits
        elif mode == 2:
            history_mask_extended = history_mask.unsqueeze(2).repeat(1,1,self.hidden_size)
            logits = self.fuse_intent(hidden_states, intent_latent, history_mask_extended)
            return logits
        else:
            logits = self.lm_head(hidden_states)
            return logits


    def decoding(self, input_ids, past=None, cls_mask=None, token_type_ids=None, mode=0, current_emotion_latent=None, 
                 future_emotion_latent=None, intent_latent=None, sigma=None):
        input_embedding = self._embed(input_ids, token_type_ids, past)
        transformer_outputs = self.transformer(input_embed=input_embedding,
                                               cls_mask=cls_mask,
                                               past=past,
                                               attention_mask=None,
                                               decoding=True)
        hidden_states, present = transformer_outputs[:2]

        if mode == 0:
            if sigma is None:
                sigma = self.gate_layer(hidden_states[:, 0]).unsqueeze(2)
            fused_emotion_logits = self.fuse_emotion(hidden_states, current_emotion_latent, future_emotion_latent)
            fused_intent_logits = self.fuse_intent(hidden_states, intent_latent)
            logits, _ = self.fuse_all(hidden_states[:, 0], fused_emotion_logits, fused_intent_logits, sigma)
            return logits, present, sigma
        elif mode == 1:
            logits = self.fuse_emotion(hidden_states, current_emotion_latent, future_emotion_latent)
            return logits, present
        else:
            logits = self.fuse_intent(hidden_states, intent_latent)
            return logits, present                                       



    