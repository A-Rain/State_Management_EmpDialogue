import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_base import TransformerEncoder


class TrsEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, pretrained_embed_weights):
        super(TrsEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embed_weights, padding_idx=0, freeze=False)
        self.position_embeddings = nn.Embedding(config.max_position, config.n_embd)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.pdrop_embed)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NLU_Trs_based(nn.Module):
    def __init__(self, config, pretrained_embed_weights, emo_trans_matrix, emo_intent_trans_matrix, planning='cat'):
        super(NLU_Trs_based, self).__init__()
        self.transformer = TransformerEncoder(d_model=config.n_embd,
                                              d_ff=4 * config.n_embd,
                                              n_head=config.n_head,
                                              n_layer=config.n_layer,
                                              dropout=config.pdrop_hidden,
                                              eps=config.layer_norm_epsilon)
        
        self.embed_layer = TrsEmbeddings(config, pretrained_embed_weights)

        # emotion transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans_mat', self.emotion_trans)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.n_embd)
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.n_embd)

        self.dropout = nn.Dropout(config.pdrop_hidden)

        self.planning = planning
        if planning == 'cat':
            self.emotion_planning = nn.Linear(config.n_embd * 2, self.emotion_label_num)
            self.intent_planning = nn.Linear(config.n_embd * 2, self.intent_label_num)
        else:
            self.emotion_planning = nn.Linear(config.n_embd, self.emotion_label_num)
            self.intent_planning = nn.Linear(config.n_embd, self.intent_label_num)
        
    def emotion_classify_current(self, cls_states: torch.Tensor):
        """
        :param cls_states: [Bsz, H], weight: [emotion_num, H]
        """
        return F.linear(cls_states, self.emotion_latent.weight)

    def emotion_classify_future(self, cls_states: torch.Tensor, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = cls_states.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emotion_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.emotion_label_num),
                                        dim=1)  # [Bsz, 1, emotion_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        prior_emotion_latent = torch.matmul(batch_trans_prob, self.emotion_latent.weight)  # [Bsz, H]
        
        # emotion planning 
        if self.planning == 'cat':
            future_emotion_logits = self.emotion_planning(torch.cat((cls_states, prior_emotion_latent), dim=-1))
        else:
            future_emotion_logits = self.emotion_planning(cls_states+prior_emotion_latent)
        
        return future_emotion_logits

    def intent_classify(self, cls_states: torch.Tensor, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = cls_states.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emo_int_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.intent_label_num),
                                        dim=1)  # [Bsz, 1, intent_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        prior_emotion_latent = torch.matmul(batch_trans_prob, self.intent_latent.weight)  # [Bsz, H]

        # emotion planning 
        if self.planning == 'cat':
            future_intent_logits = self.intent_planning(torch.cat((cls_states, prior_emotion_latent), dim=-1))
        else:
            future_intent_logits = self.intent_planning(cls_states+prior_emotion_latent)
        
        return future_intent_logits

    def forward(self, input_ids, gold_current_emotion_id=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        curr_emotion_logits = self.emotion_classify_current(cls_states)

        if forcing:
            future_emotion_logits = self.emotion_classify_future(cls_states, gold_current_emotion_id)
            intent_logits = self.intent_classify(cls_states, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            future_emotion_logits = self.emotion_classify_future(cls_states, curr_pred_emotion_id)
            intent_logits = self.intent_classify(cls_states, curr_pred_emotion_id)
        
        return curr_emotion_logits, future_emotion_logits, intent_logits

    def get_intent_pred_label(self, sig_logits):
        ind = torch.topk(sig_logits, k=1, dim=-1)[1]
        pred_id_part1 = F.one_hot(ind, num_classes=self.intent_label_num).squeeze(1)
        pred_id_part2 = (sig_logits > 0.5).float()
        return ((pred_id_part1 + pred_id_part2) > 0).float()


    def get_emotion_intent_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, gold_intent_id=None, 
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            
            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return cur_emo_latent, fut_emo_latent, intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(cls_states)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            cur_emo_latent = self.emotion_latent(curr_pred_emotion_id)
            
            future_emotion_logits = self.emotion_classify_future(cls_states, curr_pred_emotion_id)
            fut_pred_emotion_id = torch.argmax(future_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            fut_emo_latent = self.emotion_latent(fut_pred_emotion_id)

            intent_logits = self.intent_classify(cls_states, curr_pred_emotion_id)
            intent_pred_id = self.get_intent_pred_label(F.sigmoid(intent_logits))
            intent_pred_per_num = torch.sum(intent_pred_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(intent_pred_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_pred_per_num).unsqueeze(1)

            if eval_mode:
                return cur_emo_latent, fut_emo_latent, intent_latent, curr_pred_emotion_id, fut_pred_emotion_id, intent_pred_id
            else:
                return cur_emo_latent, fut_emo_latent, intent_latent


class emotion_NLU_Trs_based(nn.Module):
    def __init__(self, config, pretrained_embed_weights, emo_trans_matrix, planning='cat'):
        super(emotion_NLU_Trs_based, self).__init__()
        self.transformer = TransformerEncoder(d_model=config.n_embd,
                                              d_ff=4 * config.n_embd,
                                              n_head=config.n_head,
                                              n_layer=config.n_layer,
                                              dropout=config.pdrop_hidden,
                                              eps=config.layer_norm_epsilon)
        
        self.embed_layer = TrsEmbeddings(config, pretrained_embed_weights)

        # emotion transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans_mat', self.emotion_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.n_embd)

        self.dropout = nn.Dropout(config.pdrop_hidden)

        self.planning = planning
        if planning == 'cat':
            self.emotion_planning = nn.Linear(config.n_embd * 2, self.emotion_label_num)
        else:
            self.emotion_planning = nn.Linear(config.n_embd, self.emotion_label_num)

    def emotion_classify_current(self, cls_states: torch.Tensor):
        """
        :param cls_states: [Bsz, H], weight: [emotion_num, H]
        """
        return F.linear(cls_states, self.emotion_latent.weight)

    def emotion_classify_future(self, cls_states: torch.Tensor, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = cls_states.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emotion_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.emotion_label_num),
                                        dim=1)  # [Bsz, 1, emotion_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        prior_emotion_latent = torch.matmul(batch_trans_prob, self.emotion_latent.weight)  # [Bsz, H]
        
        # emotion planning 
        if self.planning == 'cat':
            future_emotion_logits = self.emotion_planning(torch.cat((cls_states, prior_emotion_latent), dim=-1))
        else:
            future_emotion_logits = self.emotion_planning(cls_states+prior_emotion_latent)
        
        return future_emotion_logits

    def forward(self, input_ids, gold_current_emotion_id=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        curr_emotion_logits = self.emotion_classify_current(cls_states)

        if forcing:
            future_emotion_logits = self.emotion_classify_future(cls_states, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            future_emotion_logits = self.emotion_classify_future(cls_states, curr_pred_emotion_id)
        
        return curr_emotion_logits, future_emotion_logits

    def get_emotion_intent_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, gold_intent_id=None, 
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            
            return cur_emo_latent, fut_emo_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(cls_states)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            cur_emo_latent = self.emotion_latent(curr_pred_emotion_id)
            
            future_emotion_logits = self.emotion_classify_future(cls_states, curr_pred_emotion_id)
            fut_pred_emotion_id = torch.argmax(future_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            fut_emo_latent = self.emotion_latent(fut_pred_emotion_id)

            if eval_mode:
                return cur_emo_latent, fut_emo_latent, curr_pred_emotion_id, fut_pred_emotion_id
            else:
                return cur_emo_latent, fut_emo_latent


class Intent_NLU_Trs_based(nn.Module):
    def __init__(self, config, pretrained_embed_weights, emo_intent_trans_matrix, planning='cat'):
        super(Intent_NLU_Trs_based, self).__init__()
        self.transformer = TransformerEncoder(d_model=config.n_embd,
                                              d_ff=4 * config.n_embd,
                                              n_head=config.n_head,
                                              n_layer=config.n_layer,
                                              dropout=config.pdrop_hidden,
                                              eps=config.layer_norm_epsilon)
        
        self.embed_layer = TrsEmbeddings(config, pretrained_embed_weights)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.n_embd)
        
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.n_embd)

        self.dropout = nn.Dropout(config.pdrop_hidden)

        self.planning = planning
        if planning == 'cat':
            self.intent_planning = nn.Linear(config.n_embd * 2, self.intent_label_num)
        else:
            self.intent_planning = nn.Linear(config.n_embd, self.intent_label_num)
    

    def emotion_classify_current(self, cls_states: torch.Tensor):
        """
        :param cls_states: [Bsz, H], weight: [emotion_num, H]
        """
        return F.linear(cls_states, self.emotion_latent.weight)

    def get_intent_pred_label(self, sig_logits):
        ind = torch.topk(sig_logits, k=1, dim=-1)[1]
        pred_id_part1 = F.one_hot(ind, num_classes=self.intent_label_num).squeeze(1)
        pred_id_part2 = (sig_logits > 0.5).float()
        return ((pred_id_part1 + pred_id_part2) > 0).float()

    def intent_classify(self, cls_states: torch.Tensor, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = cls_states.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emo_int_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.intent_label_num),
                                        dim=1)  # [Bsz, 1, intent_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        prior_emotion_latent = torch.matmul(batch_trans_prob, self.intent_latent.weight)  # [Bsz, H]

        # emotion planning 
        if self.planning == 'cat':
            future_intent_logits = self.intent_planning(torch.cat((cls_states, prior_emotion_latent), dim=-1))
        else:
            future_intent_logits = self.intent_planning(cls_states+prior_emotion_latent)
        
        return future_intent_logits
    
    def forward(self, input_ids, gold_current_emotion_id=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        curr_emotion_logits = self.emotion_classify_current(cls_states)

        if forcing:
            intent_logits = self.intent_classify(cls_states, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_logits = self.intent_classify(cls_states, curr_pred_emotion_id)
        
        return curr_emotion_logits, intent_logits


    def get_intent_latent(self, input_ids, gold_intent_id=None, attention_mask=None, token_type_ids=None,
                          position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        if forcing:
            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(cls_states)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]

            intent_logits = self.intent_classify(cls_states, curr_pred_emotion_id)
            intent_pred_id = self.get_intent_pred_label(F.sigmoid(intent_logits))
            intent_pred_per_num = torch.sum(intent_pred_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(intent_pred_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_pred_per_num).unsqueeze(1)

            if eval_mode:
                return intent_latent, curr_pred_emotion_id, intent_pred_id
            else:
                return intent_latent


class NLU_Trs_based_naive(nn.Module):
    def __init__(self, config, pretrained_embed_weights, emo_trans_matrix, emo_intent_trans_matrix, planning='cat'):
        super(NLU_Trs_based_naive, self).__init__()
        self.transformer = TransformerEncoder(d_model=config.n_embd,
                                              d_ff=4 * config.n_embd,
                                              n_head=config.n_head,
                                              n_layer=config.n_layer,
                                              dropout=config.pdrop_hidden,
                                              eps=config.layer_norm_epsilon)
        
        self.embed_layer = TrsEmbeddings(config, pretrained_embed_weights)

        # emotion transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans_mat', self.emotion_trans)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.n_embd)
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.n_embd)

        self.dropout = nn.Dropout(config.pdrop_hidden)

        
    def emotion_classify_current(self, cls_states: torch.Tensor):
        """
        :param cls_states: [Bsz, H], weight: [emotion_num, H]
        """
        return F.linear(cls_states, self.emotion_latent.weight)

    def emotion_classify_future(self, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = pred_or_true_current_emo_id.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emotion_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.emotion_label_num),
                                        dim=1)  # [Bsz, 1, emotion_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        
        return batch_trans_prob

    def intent_classify(self, pred_or_true_current_emo_id: torch.Tensor):
        Bsz = pred_or_true_current_emo_id.shape[0]

        # get prior emotion latent
        batch_trans_prob = torch.gather(self.emo_int_trans.unsqueeze(0).repeat(Bsz, 1, 1),
                                        index=pred_or_true_current_emo_id.unsqueeze(2).repeat(1, 1, self.intent_label_num),
                                        dim=1)  # [Bsz, 1, intent_label_num]
        batch_trans_prob = batch_trans_prob.squeeze(1)  
        
        return batch_trans_prob

    def forward(self, input_ids, gold_current_emotion_id=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]
        cls_states = outputs[:, 0]
        curr_emotion_logits = self.emotion_classify_current(cls_states)
        
        return curr_emotion_logits

    def get_emotion_intent_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, gold_intent_id=None, 
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            
            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return cur_emo_latent, fut_emo_latent, intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(cls_states)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            cur_emo_latent = self.emotion_latent(curr_pred_emotion_id)
            
            fut_emotion_prob = self.emotion_classify_future(curr_pred_emotion_id)
            fut_emotion_pred_id = torch.argmax(fut_emotion_prob, dim=-1).unsqueeze(1)
            fut_emo_latent = self.emotion_latent(fut_emotion_pred_id)

            intent_prob = self.intent_classify(curr_pred_emotion_id)
            intent_pred_id = torch.argmax(intent_prob, dim=-1).unsqueeze(1)
            intent_latent = self.intent_latent(intent_pred_id)

            if eval_mode:
                return cur_emo_latent, fut_emo_latent, intent_latent, curr_pred_emotion_id, fut_emotion_pred_id, intent_pred_id
            else:
                return cur_emo_latent, fut_emo_latent, intent_latent

    def predict(self, input_ids, gold_current_emotion_id=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        input_embed = self.embed_layer(input_ids)
        outputs = self.transformer(input_embed, attention_mask)
        outputs = self.dropout(outputs)  # [Bsz, L, H]

        cls_states = outputs[:, 0]

        curr_emotion_logits = self.emotion_classify_current(cls_states)
        curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1)
        future_emotion_logits = self.emotion_classify_future(curr_pred_emotion_id.unsqueeze(1))
        fut_pred_emotion_id = torch.argmax(future_emotion_logits, dim=-1)
        intent_logits = self.intent_classify(curr_pred_emotion_id.unsqueeze(1))
        intent_pred_id = torch.argmax(intent_logits, dim=-1)

        return curr_pred_emotion_id, fut_pred_emotion_id, intent_pred_id

