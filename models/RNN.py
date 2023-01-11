import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, rnntype, bidirect, dropout, input_dim, hidden_dim, n_layers, n_classes, device, firstBN):
        super().__init__()
        self.firstBN = firstBN
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rnntype = rnntype
        self.bn1 = nn.BatchNorm1d(input_dim * input_dim)
        if self.rnntype == 'LSTM':
            self.rnn = nn.LSTM(input_dim,hidden_dim,n_layers,batch_first=True,device=device,bidirectional=bidirect,dropout=dropout)
        elif self.rnntype == 'GRU':
            self.rnn = nn.GRU(input_dim,hidden_dim,n_layers,batch_first=True,device=device,bidirectional=bidirect,dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, din):
        x = din
        if self.firstBN:
            a,b,c = x.size()
            x = x.view(a,-1)
            x = self.bn1(x)
            x = x.view(a,b,c)
        if self.rnntype == 'LSTM':
            out, (h_n,c_n) = self.rnn(x)
            x = h_n[-1,:,:]            
        elif self.rnntype == 'GRU':
            out, h_n = self.rnn(x)
            x = h_n[-1,:,:]
        x = self.bn2(x)
        dout = self.classifier(x)
        # dout = F.log_softmax(dout)
        # dout = F.softmax(dout)
        return dout
        