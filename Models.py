import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                n_layers, hidden_dim, bidirectional, dropout):
        self.bidirectional = bidirectional
        super().__init__()
        self.id = 'clstm'
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.rnn = nn.LSTM(1, 
                            hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)
                            
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, lyrics):
        embedded = self.embedding(lyrics)
        embedded = embedded.unsqueeze(1)        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
                
        cat = self.dropout(torch.cat(pooled, dim = 1))
        cat_exp = cat.unsqueeze(-1).permute(1, 0, 2)
        output, (hidden, cell) = self.rnn(cat_exp)

        if self.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        

        return self.fc(hidden)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
        self.id = 'cnn'
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lyrics):
        embedded = self.embedding(lyrics)
        embedded = embedded.unsqueeze(1)        
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]       
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
    bidirectional, dropout, pad_idx):
        super().__init__()
        self.id = 'lstm'
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
        hidden_dim,
        num_layers=n_layers,
        bidirectional=bidirectional,
        dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embed = self.dropout(self.embedding(text))
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embed)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

