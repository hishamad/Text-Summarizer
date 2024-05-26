import torch


class Config:
    def __init__(self):
        self.PADDING_SYMBOL = ' '
        self.START_SYMBOL = '<START>'
        self.END_SYMBOL = '<END>'
        self.UNK_SYMBOL = '<UNK>'
        self.MAX_PREDICTIONS = 20
        
        self.use_attention = True  
        self.use_gru = True
        self.bidirectional = True
        
        self.use_embeddings = True
        self.tune_embeddings = True
        
        self.hidden_size = 25
        self.learning_rate = 0.01
        
        self.batch_size = 64
        self.epochs = 40
        
        self.save = False
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'