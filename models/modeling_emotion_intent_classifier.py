from models.pytorch_transformers import BertPreTrainedModel, BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class Emotion_classifier(BertPreTrainedModel):
    def __init__(self, config, emo_trans_matrix, planning='cat'):
        super(Emotion_classifier, self).__init__(config)
        self.bert = BertModel(config)

        # transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans', self.emotion_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.planning = planning
        if planning == 'cat':
            self.emotion_planning = nn.Linear(config.hidden_size * 2, self.emotion_label_num)
        else:
            self.emotion_planning = nn.Linear(config.hidden_size, self.emotion_label_num)

        self.init_weights()
    
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
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        curr_emotion_logits = self.emotion_classify_current(pooled_output)

        if forcing:
            future_emotion_logits = self.emotion_classify_future(pooled_output, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            future_emotion_logits = self.emotion_classify_future(pooled_output, curr_pred_emotion_id)
        
        return curr_emotion_logits, future_emotion_logits

    def fuse_current_future(self, current_emotion_id, future_emotion_id):
        """
        current_emotion_id: [Bsz]
        future_emotion_id: [Bsz, 1]
        """
        row = torch.index_select(self.emotion_trans, dim=0, index=current_emotion_id)
        col = torch.gather(row, dim=1, index=future_emotion_id)  # [Bsz, 1]
        return col

    def get_emotion_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, forcing=False, eval_mode=False,
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            return cur_emo_latent, fut_emo_latent
            # trans_prob = self.fuse_current_future(gold_current_emotion_id.squeeze(1), gold_future_emotion_id)
            # trans_prob = trans_prob.unsqueeze(2)
            # fused_latent = trans_prob * cur_emo_latent + (1 - trans_prob) * fut_emo_latent
            # return fused_latent, fused_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(pooled_output)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            cur_emo_latent = self.emotion_latent(curr_pred_emotion_id)
            
            future_emotion_logits = self.emotion_classify_future(pooled_output, curr_pred_emotion_id)
            fut_pred_emotion_id = torch.argmax(future_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            fut_emo_latent = self.emotion_latent(fut_pred_emotion_id)

            # trans_prob = self.fuse_current_future(curr_pred_emotion_id.squeeze(1), fut_pred_emotion_id)
            # trans_prob = trans_prob.unsqueeze(2)
            # fused_latent = trans_prob * cur_emo_latent + (1 - trans_prob) * fut_emo_latent

            if eval_mode:
                return cur_emo_latent, fut_emo_latent, curr_pred_emotion_id, fut_pred_emotion_id
            else:
                return cur_emo_latent, fut_emo_latent


class Intent_classifier(BertPreTrainedModel):
    def __init__(self, config, emo_intent_trans_matrix, planning='cat'):
        super(Intent_classifier, self).__init__(config)
        self.bert = BertModel(config)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.hidden_size)
        
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.planning = planning
        if planning == 'cat':
            self.intent_planning = nn.Linear(config.hidden_size * 2, self.intent_label_num)
        else:
            self.intent_planning = nn.Linear(config.hidden_size, self.intent_label_num)

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
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        curr_emotion_logits = self.emotion_classify_current(pooled_output)

        if forcing:
            intent_logits = self.intent_classify(pooled_output, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_logits = self.intent_classify(pooled_output, curr_pred_emotion_id)
        
        return curr_emotion_logits, intent_logits

    def get_intent_latent(self, input_ids, gold_intent_id=None, forcing=False, eval_mode=False,
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if forcing:
            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(pooled_output)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]

            intent_logits = self.intent_classify(pooled_output, curr_pred_emotion_id)
            intent_pred_id = self.get_intent_pred_label(F.sigmoid(intent_logits))
            intent_pred_per_num = torch.sum(intent_pred_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(intent_pred_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_pred_per_num).unsqueeze(1)

            if eval_mode:
                return intent_latent, curr_pred_emotion_id, intent_pred_id
            else:
                return intent_latent



class emotion_intent_classifier(BertPreTrainedModel):
    def __init__(self, config, emo_trans_matrix, emo_intent_trans_matrix, planning='cat'):
        super(emotion_intent_classifier, self).__init__(config)
        self.bert = BertModel(config)

        # emotion transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans_mat', self.emotion_trans)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.hidden_size)
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.planning = planning
        if planning == 'cat':
            self.emotion_planning = nn.Linear(config.hidden_size * 2, self.emotion_label_num)
            self.intent_planning = nn.Linear(config.hidden_size * 2, self.intent_label_num)
        else:
            self.emotion_planning = nn.Linear(config.hidden_size, self.emotion_label_num)
            self.intent_planning = nn.Linear(config.hidden_size, self.intent_label_num)
        
    def emotion_classify_current(self, cls_states: torch.Tensor):
        """
        :param cls_states: [Bsz, H], weight: [emotion_num, H]
        """
        return F.linear(cls_states, self.emotion_latent.weight)

    def fuse_current_future(self, current_emotion_id, future_emotion_id):
        """
        current_emotion_id: [Bsz]
        future_emotion_id: [Bsz, 1]
        """
        row = torch.index_select(self.emotion_trans, dim=0, index=current_emotion_id)
        col = torch.gather(row, dim=1, index=future_emotion_id)  # [Bsz, 1]
        return col

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
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        curr_emotion_logits = self.emotion_classify_current(pooled_output)

        if forcing:
            future_emotion_logits = self.emotion_classify_future(pooled_output, gold_current_emotion_id)
            intent_logits = self.intent_classify(pooled_output, gold_current_emotion_id)
        else:
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            future_emotion_logits = self.emotion_classify_future(pooled_output, curr_pred_emotion_id)
            intent_logits = self.intent_classify(pooled_output, curr_pred_emotion_id)
        
        return curr_emotion_logits, future_emotion_logits, intent_logits


    def get_intent_pred_label(self, sig_logits):
        ind = torch.topk(sig_logits, k=1, dim=-1)[1]
        pred_id_part1 = F.one_hot(ind, num_classes=self.intent_label_num).squeeze(1)
        pred_id_part2 = (sig_logits > 0.5).float()
        return ((pred_id_part1 + pred_id_part2) > 0).float()


    def get_emotion_intent_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, gold_intent_id=None, 
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            # trans_prob = self.fuse_current_future(gold_current_emotion_id.squeeze(1), gold_future_emotion_id)
            # trans_prob = trans_prob.unsqueeze(2)
            # fused_latent = trans_prob * cur_emo_latent + (1 - trans_prob) * fut_emo_latent

            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return cur_emo_latent, fut_emo_latent, intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(pooled_output)
            curr_pred_emotion_id = torch.argmax(curr_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            cur_emo_latent = self.emotion_latent(curr_pred_emotion_id)
            
            future_emotion_logits = self.emotion_classify_future(pooled_output, curr_pred_emotion_id)
            fut_pred_emotion_id = torch.argmax(future_emotion_logits, dim=-1).unsqueeze(1)  # [Bsz, 1]
            fut_emo_latent = self.emotion_latent(fut_pred_emotion_id)

            intent_logits = self.intent_classify(pooled_output, curr_pred_emotion_id)
            intent_pred_id = self.get_intent_pred_label(F.sigmoid(intent_logits))
            intent_pred_per_num = torch.sum(intent_pred_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(intent_pred_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_pred_per_num).unsqueeze(1)

            # trans_prob = self.fuse_current_future(curr_pred_emotion_id.squeeze(1), fut_pred_emotion_id)
            # trans_prob = trans_prob.unsqueeze(2)
            # fused_latent = trans_prob * cur_emo_latent + (1 - trans_prob) * fut_emo_latent

            if eval_mode:
                return cur_emo_latent, fut_emo_latent, intent_latent, curr_pred_emotion_id, fut_pred_emotion_id, intent_pred_id
            else:
                return cur_emo_latent, fut_emo_latent, intent_latent


class emotion_intent_classifier_naive(BertPreTrainedModel):
    def __init__(self, config, emo_trans_matrix, emo_intent_trans_matrix, planning='cat'):
        super(emotion_intent_classifier_naive, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # emotion transition matrix
        self.emotion_trans = emo_trans_matrix
        self.register_buffer('emo_trans_mat', self.emotion_trans)

        # emotion-intent transition matrix
        self.emo_int_trans = emo_intent_trans_matrix
        self.register_buffer('emo_int_trans_mat', self.emo_int_trans)

        self.emotion_label_num = config.num_labels_emo
        self.emotion_latent = nn.Embedding(config.num_labels_emo, config.hidden_size)
        self.intent_label_num = config.num_labels_intent
        self.intent_latent = nn.Embedding(config.num_labels_intent, config.hidden_size)

        
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
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        curr_emotion_logits = self.emotion_classify_current(pooled_output)
        
        return curr_emotion_logits

    def get_emotion_intent_latent(self, input_ids, gold_current_emotion_id=None, gold_future_emotion_id=None, gold_intent_id=None, 
                                  attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False, eval_mode=False):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if forcing:
            cur_emo_latent = self.emotion_latent(gold_current_emotion_id)
            fut_emo_latent = self.emotion_latent(gold_future_emotion_id)
            # trans_prob = self.fuse_current_future(gold_current_emotion_id.squeeze(1), gold_future_emotion_id)
            # trans_prob = trans_prob.unsqueeze(2)
            # fused_latent = trans_prob * cur_emo_latent + (1 - trans_prob) * fut_emo_latent

            intent_per_num = torch.sum(gold_intent_id, dim=-1).unsqueeze(1)  # [Bsz, 1]
            intent_latent = torch.matmul(gold_intent_id, self.intent_latent.weight)  # [Bsz, H]    
            intent_latent = (intent_latent / intent_per_num).unsqueeze(1)
            return cur_emo_latent, fut_emo_latent, intent_latent
        else:
            curr_emotion_logits = self.emotion_classify_current(pooled_output)
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

    def predict(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, forcing=False):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        curr_emotion_logits = self.emotion_classify_current(pooled_output)
        curr_emotion_pred_id = torch.argmax(curr_emotion_logits, dim=-1)

        fut_emotion_prob = self.emotion_classify_future(curr_emotion_pred_id.unsqueeze(1))
        fut_emotion_pred_id = torch.argmax(fut_emotion_prob, dim=-1)
        intent_prob = self.intent_classify(curr_emotion_pred_id.unsqueeze(1))
        intent_pred_id = torch.argmax(intent_prob, dim=-1)

        return curr_emotion_pred_id, fut_emotion_pred_id, intent_pred_id