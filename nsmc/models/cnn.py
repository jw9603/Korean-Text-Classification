import torch
import torch.nn as nn
class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        word_vec_size,
        n_classes,
        use_batch_norm=False,
        dropout_p=.5,
        window_sizes=[3, 4, 5],
        n_filters=[100, 100, 100],
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size, word_vec_size)
        
        self.feature_extractors = nn.ModuleList()
        for ws, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=n_filter,
                        kernel_size=(ws, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p)
                )
            )
        
        self.generator = nn.Linear(sum(n_filters), n_classes)
        self.act = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        # |x| = (bs, length)
        x = self.emb(x)
        # |x| = (bs, length, ws)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            pad = x.new(x.size(0), min_length - x.size(1), self.word_vec_size).zero_()
            # |pad| = (bs, min_length - length, word_vec_size)
            x = torch.cat([x, pad],dim=1)
            # |x| = (bs, min_length, word_vec_size)
        
        x = x.unsqueeze(1)
        # |x| = (bs, 1, min_length, word_vec_size)
        
        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x)
            # |cnn_out| = (bs, n_filter, length- window_size + 1, 1)
            
            # Maxpooling References: https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html
            # input's shape : (bs, channel, L)
            cnn_out = nn.functional.max_pool1d( 
                input=cnn_out.squeeze(-1), 
                kernel_size=cnn_out.size(-2)
            ).squeeze(-1) # |cnn_out| = (bs, n_filter)
            cnn_outs.append(cnn_out)
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        # |cnn_outs| = (bs, sum(n_filters))
        y = self.act(self.generator(cnn_outs))
        # |y| = (bs, n_classes)
        return y