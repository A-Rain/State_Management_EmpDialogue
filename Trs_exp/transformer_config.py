class NLG_config(object):
    def __init__(self):
        self.layer_norm_epsilon = 1e-05
        self.n_ctx = 512
        self.n_embd = 300
        self.n_head = 6
        self.n_layer = 6
        self.max_position = 512
        self.vocab_size = 20000
        self.type_num = 2
        self.resid_pdrop=0.2
        self.embd_pdrop=0.2
        self.attn_pdrop=0.2
        self.output_attentions = False

class NLU_config(object):
    def __init__(self):
        self.n_embd = 300
        self.n_head = 5
        self.n_layer = 4
        self.pdrop_hidden = 0.1
        self.pdrop_embed = 0.2
        self.layer_norm_epsilon = 1e-05
        self.num_labels_emo = 7
        self.num_labels_intent = 9
        self.max_position = 512
        self.type_vocab_size = 2
        self.vocab_size = 20000