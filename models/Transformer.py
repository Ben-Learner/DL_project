import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_classes, nhid, nlayers, sql):
        super(Transformer, self).__init__()
        self.d_model = nhid
        self.sql = sql
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=2, dim_feedforward=nhid, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.drop = nn.Dropout(0.2)
        self.decoder = nn.Linear(nhid, num_classes)
        self.init_weights()
        self.pe = self.position_embed().cuda()
        self.bn = nn.BatchNorm1d(9216)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.cuda()

    # 位置编码层
    def position_embed(self):
        seq_length = self.sql
        position_encodings = torch.zeros((seq_length, 1, self.d_model))
        for pos in range(seq_length):
            for i in range(self.d_model):
                position_encodings[pos, 0, i] = pos / math.pow(10000, (i - i % 2) / self.d_model)
        position_encodings[:, 0, 0::2] = torch.sin(position_encodings[:, 0, 0::2])  # 2i
        position_encodings[:, 0, 1::2] = torch.cos(position_encodings[:, 0, 1::2])  # 2i+1
        return position_encodings


    def init_weights(self):
        init_uniform = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        # input:32*96*96
        a,b,c = input.size()
        input = input.view(a,-1)
        input = self.bn(input)
        input = input.view(a,b,c).transpose(0,1)
        #input:96*32*96
        input = input*math.sqrt(self.d_model)     
        input = input + self.pe
        mask = self.generate_square_subsequent_mask(self.sql)
        output= self.transformer_encoder(input, mask=mask)
        output = self.decoder(output)
        output = output.mean(dim=0)
        # output = F.log_softmax(output )
        return output