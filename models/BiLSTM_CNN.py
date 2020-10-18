# @Author : damioonfan
# @Datetime : 2020/10/10 16:06
# @File : BiLSTM.py
# @Last Modify Time : 2020/10/10 16:06
# @Contact : damionfan@163.com

"""
    FILE :  BiLSTM.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from models.initialize import *
from DataUtils.Common import *
from torch.nn import init
from models.modelHelp import prepare_pack_padded_sequence
torch.manual_seed(seed_num)
random.seed(seed_num)



import math


class Attention(nn.Module):
    """
        Attention
    """

    def __init__(self,input_dim,head_num):
        super(Attention,self).__init__()
        

        self.input_dim = input_dim

        self.q_linear = nn.Linear(in_features=self.input_dim, out_features=head_num*self.input_dim, bias=True)
        init_linear_weight_bias(self.q_linear)

        self.k_linear = nn.Linear(in_features=self.input_dim, out_features=head_num*self.input_dim, bias=True)
        init_linear_weight_bias(self.q_linear)
        self.v_linear = nn.Linear(in_features=self.input_dim, out_features=head_num*self.input_dim, bias=True)
        init_linear_weight_bias(self.q_linear)


    def forward(self,inputs):
        """
        input_dim:  [bs, ml*ml_c, 2*feature_dim]
        """
        k = self.k_linear(inputs)
        q = self.q_linear(inputs)
        v = self.v_linear(inputs)

        content = torch.bmm(q,k.permute(0,2,1))/math.sqrt(k.size(-1))
        content = torch.nn.functional.softmax(content,dim=-1)


        return torch.matmul(content,v) #[bs, ml*ml_c, feature_dim]

class Cross_Attention(nn.Module):
    """
        Cross Attention
    """
    def __init__(self, dim1,dim2,hidden_dim):
        super(Cross_Attention,self).__init__()
        # cross attention
        self.f1_bigru = nn.GRU(input_size=dim1, hidden_size=hidden_dim//2, num_layers=1,bidirectional=True, batch_first=True, bias=True)
        self.f2_bigru = nn.GRU(input_size=dim2, hidden_size=hidden_dim//2, num_layers=1,bidirectional=True, batch_first=True, bias=True)
    
    def forward(self,f1,f2):
        f1,_ = self.f1_bigru(f1)
        #f1 = torch.tanh(f1)
        f2,_ = self.f2_bigru(f2)
        #f2 = torch.tanh(f2)

        m1 = torch.bmm(f1,f2.permute(0,2,1))
        m2 = torch.bmm(f2,f1.permute(0,2,1))

        n1 = torch.nn.functional.softmax(m1,dim=-1)
        n2 = torch.nn.functional.softmax(m2,dim=-1)

        o1 = torch.bmm(n1,f2)
        o2 = torch.bmm(n2,f1)

        a1 = torch.mul(o1,f1)
        a2 = torch.mul(o2,f2)

        return torch.cat((a1,a2),dim=-1)
        #return a2


class BiLSTM_CNN(nn.Module):
    """
        BiLSTM_CNN
    """

    def __init__(self, **kwargs):
        super(BiLSTM_CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        char_paddingId = self.char_paddingId

        # word embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)

        # char embedding layer
        self.char_embedding = nn.Embedding(self.char_embed_num, self.char_dim, padding_idx=char_paddingId)
        # init_embedding(self.char_embedding.weight)
        init_embed(self.char_embedding.weight)

        # dropout
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        #attention headnumber
        self.head_num = 60
        self.word_head = 60

        #cross_attenrtion_paramater
        self.hidden_dim = 2
        self.word_hidden_dim = 2

        # cnn
        # self.char_encoders = nn.ModuleList()
        self.char_encoders = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.conv_filter_nums[i], kernel_size=(1, filter_size, self.char_dim))
            self.char_encoders.append(f)
        for conv in self.char_encoders:
            if self.device != cpu_device:
                conv.cuda()



        lstm_input_dim = D  +4*self.hidden_dim +2*self.word_hidden_dim+ sum(self.conv_filter_nums)
        self.bilstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=C, bias=True)
        init_linear_weight_bias(self.linear)
        
        # global attention 
        self.global_bigru = nn.GRU(input_size=self.char_dim, hidden_size=self.char_dim, num_layers=self.lstm_layers,bidirectional=True, batch_first=True, bias=True)
        self.char_attention = Attention(2*self.char_dim,head_num=self.head_num)
        
        #local attention
        self.word_local = Attention(D,head_num=self.word_head)
        self.word_local_bilstm = nn.LSTM(input_size=D, hidden_size=D//2, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        self.char_local_att = Attention(self.char_dim,head_num=self.head_num)


        # cross_attention
        #self.ca1 = Cross_Attention(D,sum(self.conv_filter_nums),self.hidden_dim) # cnn
        self.ca2 = Cross_Attention(sum(self.conv_filter_nums),self.head_num,self.hidden_dim) # char_local
        self.ca3 = Cross_Attention(sum(self.conv_filter_nums),self.head_num,self.hidden_dim) # char_global

        self.ca4 = Cross_Attention(sum(self.conv_filter_nums),self.word_head,self.word_hidden_dim) # char_global


    def _char_global_attention(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]

        char_content_embed,_ = self.global_bigru(input_embed) # [bs, ml*ml_c, 2*char_dim]
        char_content_embed = torch.tanh(char_content_embed)
        char_attention = self.char_attention(char_content_embed) # [bs, ml*ml_c, 2*char_dim]
        #print(char_attention.size())
        attention_out = char_attention.view(-1, max_len, max_len_char,self.head_num ,2*self.char_dim) # [bs, ml,ml_c,head ,2*char_dim]
        #print(attention_out.size())
        pool_output = torch.max(torch.sum(attention_out, -1)/2*self.char_dim,-2)[0] # [bs, ml, self.conv_filter_nums[0]]
        #pool_output = pool_output.permute(0, 2, 1)
        return pool_output

    def _char_forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # input_embed = self.dropout_embed(input_embed)
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.char_dim)
        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)

        return char_conv_outputs

    def char_local_attention(self,inputs):
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # input_embed = self.dropout_embed(input_embed)
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, max_len*max_len_char, self.char_dim)
        attention = self.char_local_att(input_embed)
        attention = attention.view(-1, max_len, max_len_char, self.head_num,self.char_dim)
        attention = torch.max(torch.sum(attention,dim=-1)/self.char_dim,dim=-2)[0]

        return attention

    
    def word_local_attention(self,word):
        max_len = word.size(1)
        word,_ = self.word_local_bilstm(word)
        word = torch.tanh(word)
        x = self.word_local(word)
        #print(x.size())
        attention_out = x.view(-1,max_len,self.word_head,self.embed_dim)
        #print(attention_out.size())
        pool_output = torch.max(attention_out,-1)[0]
        return pool_output
    

    def forward(self, word, char, sentence_length):
        """
        :param char:
        :param word:
        :param sentence_length:
        :return:
        """
        char_conv = self._char_forward(char)
        char_conv = self.dropout(char_conv)

        char_local_attention = self.char_local_attention(char)
        char_local_attention = self.dropout(char_local_attention)
        char_attention = self._char_global_attention(char)
        char_attention = self.dropout(char_attention) # [bs, ml, 2*char_dim]
        
        word = self.embed(word)  # (N,W,D)
        word_local = self.word_local_attention(word)
        word_local = self.dropout(word_local)
        
        word_embed = self.ca4(char_conv,word_local)

        #char_conv1 = self.ca1(word,char_conv)
        char_local_attention = self.ca2(char_conv,char_local_attention)
        char_attention = self.ca3(char_conv,char_attention)

        x = torch.cat((word,word_embed,char_local_attention,char_attention,char_conv), -1)        

        x = self.dropout_embed(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit


"""
Clustering of missense mutations in the ataxia - telangiectasia in a sporadic T - cell leukaemia .
"""
