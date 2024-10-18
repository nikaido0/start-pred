#!/usr/bin/env python
#coding=gbk

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.ban import BANLayer
from models.FFN import FeatureFusionNetwork

class TextCNN_block1(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block1, self).__init__()
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim_DLM,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])  # ���֦���
        self.fc1 = nn.Linear(1920, 512)
        self.fc = nn.Sequential(
            nn.Linear(512, 32),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(32, 8),
            nn.Mish(),
            nn.Linear(8, output_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        self.batchnorm1 = nn.BatchNorm1d(512)


    def forward(self, DLM_fea, seq_data, chr_data):
        # ���������ݽ���ά�ȱ任
        # ���������ݵ�ά�ȴ� [batch_size, sequence_length, embedding_dim] ��Ϊ [batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 2, 1)
        # Ӧ�þ���㲢�ϲ����
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1] # nn.Conv1d ��Ҫ���� [batch_size, channels, sequence_length] ��״������

        # �ػ���
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]

        # ���֧����չ��
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]

        # ������֧������һ��
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))  ##���֧���Ӻ�ά��Ϊ��n_filters * filter_sizes * 10
        DLM_cat_i = self.fc1(DLM_cat)   #ʹ�����Բ����ά�ȱ任
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # �������������
        return self.fc(DLM_cat_i), DLM_cat_i


class TextCNN_block2(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN_block2, self).__init__()
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=4,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])  # ���֦���
        self.fc2 = nn.Linear(1920, 512)
        self.fc6 = nn.Linear(919, 512)
        self.fc3 = nn.Sequential(
            nn.Linear(512, 32),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(32, 8),
            nn.Mish(),
            nn.Linear(8, output_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(919)
        self.ban1 = BANLayer(512, 512, 512, 2, 0.1, 6)  # Bilinear Attention Networks


    def forward(self, DLM_fea, seq_data, chr_data):
        # ���������ݽ��д�����ӳ��
#        seq_embedded = self.embedding(seq_data)
        # ���������ݽ���ά�ȱ任
        seq_embedded2 = seq_data.permute(0, 2, 1)
        # Ӧ�þ���㲢�ϲ����
        seq_conved = [self.Mish2(conv(seq_embedded2)) for conv in self.convs2] # nn.Conv1d ��Ҫ���� [batch_size, channels, sequence_length] ��״������
        # �ػ���
        seq_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in seq_conved]
        # ���֧����չ��
        seq_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in seq_pooled]
        # ������֧������һ��
        seq_cat = self.dropout2(torch.cat(seq_flatten, dim=1))  ##���֧���Ӻ�ά��Ϊ��n_filters * filter_sizes * 10
        seq_cat_i = self.fc2(seq_cat)   #ʹ�����Բ����ά�ȱ任
        seq_cat_i = self.batchnorm2(seq_cat_i)

        # ����Ⱦɫ���������
        fea_data = self.batchnorm4(chr_data)
        fea = self.fc6(fea_data)   #ʹ�����Բ����ά�ȱ任

        # ���������ݺ�Ⱦɫ����������ں���һ��
        fusion, att_weight = self.ban1(seq_cat_i.unsqueeze(1), fea.unsqueeze(1))
       
        # �������������
        return self.fc3(fusion), fusion
        


class StartPred(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters, filter_sizes, output_dim, dropout):
        super(StartPred, self).__init__()

        self.DLM_encoder = TextCNN_block1(vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters, filter_sizes, output_dim, dropout)
        self.seq_encoder1 = TextCNN_block2(vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters, filter_sizes, output_dim, dropout)

        self.fc5 = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Linear(64, output_dim)
        )

        self.batchnorm3 = nn.BatchNorm1d(1024)
        

    def forward(self, DLM_fea, seq_data, chr_data):
        # ���������ݽ��д���
        _, data1 = self.DLM_encoder(DLM_fea, seq_data, chr_data) #GPN-MSA��TextCNN������
        _, data2 = self.seq_encoder1(DLM_fea, seq_data, chr_data) #ͻ�����м�����Ŵ����ε��ں�����


        # �������������ƴ��
        fea = torch.cat([data1, data2], dim=1)
        fea = self.batchnorm3(fea)


        # �������������
        return self.fc5(fea), fea