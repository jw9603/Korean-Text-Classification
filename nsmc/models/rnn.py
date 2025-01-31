import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super().__init__()
        # embedding references: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.emb = nn.Embedding(input_size, word_vec_size)
        
        # rnn references: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        self.act = nn.LogSoftmax(dim=-1)
    
    def forward(self,x):
        # |x| = (bs, length)
        x = self.emb(x)
        # |x| = (bs, length, ws)
        x, _ = self.rnn(x)
        # |x| = (bs, length, hs*2)
        y = self.act(self.generator(x[:,-1]))
        # |y| = (bs, n_classes)
        return y