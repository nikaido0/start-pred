#!/usr/bin/env python
# coding=gbk

import argparse


def get_config():
    parse = argparse.ArgumentParser(description='default config')
    parse.add_argument('-a', type=str, default='model')
    # 数据参数
    parse.add_argument('-vocab_size', type=int, default=4, help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=1, help='Number of mutation functions')
    parse.add_argument('-CV', type=bool, default=False, help='Cross validation')

    # 训练参数
    parse.add_argument('-batch_size', type=int, default=128, help='Batch size')
    parse.add_argument('-epochs', type=int, default=200)
    parse.add_argument('-learning_rate', type=float, default=0.0001)
    parse.add_argument('-threshold', type=float, default=0.5)
    parse.add_argument('-early_stop', type=int, default=10)

    # 模型参数
    parse.add_argument('-model_name', type=str, default='StartPred', help='Name of the model')
    parse.add_argument('-embedding_size_DLM', type=int, default=768, help='Dimension of the embedding') # DNA语言模型特征维度
    parse.add_argument('-DLM_seq_len', type=int, default=128, help='Length of the sequence in DLM model') # DNA语言模型序列长度
    parse.add_argument('-embedding_size_seq', type=int, default=128, help='Dimension of the embedding') # 序列相关的特征维度
    parse.add_argument('-sequence_length', type=int, default=1001, help='Length of the mutation sequence') # 突变序列长度
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64, help='Number of the filter') # 卷积核数量
    parse.add_argument('-filter_size', type=list, default=[3, 4, 5], help='Size of the filter') # 卷积核大小

    
    # 路径参数
    ## Ref-seq feature of training set
    parse.add_argument('-train_direction', type=str, default='../data/train_GPN-MSA_feature.pth', help='The ref-seq feature of training set')
    ## Mut-seq of training set
    parse.add_argument('-train_label_direction', type=str, default='../data/train.csv', help='The Mut-seq of training set')
    ## Epigenetic feature of training set
    parse.add_argument('-chrom_train_direction', type=str, default='../data/train_DanQ_feature.h5', help='The epigenetic feature of training set')
    ## Ref-seq feature of test set
    parse.add_argument('-test_direction', type=str, default='../data/test_GPN-MSA_feature.pth', help='The ref-seq feature of test set')
    ## Mut-seq of test set
    parse.add_argument('-test_label_direction', type=str, default='../data/test.csv', help='The Mut-seq of test set')
    ## Epigenetic feature of test set
    parse.add_argument('-chrom_test_direction', type=str, default='../data/test_DanQ_feature.h5', help='The epigenetic feature of test set')

    config = parse.parse_args()
    return config
