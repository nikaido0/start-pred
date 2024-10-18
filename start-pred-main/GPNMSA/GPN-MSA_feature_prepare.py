#!/usr/bin/env python
#coding=gbk

from gpn.data import GenomeMSA, Tokenizer
import gpn.model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM

# ����ģ��·���Ͷ����бȶԣ�MSA������·��
model_path = './model'
msa_path = './89.zarr.zip'

# ��ȡ����
test_data = pd.read_csv("../data/test.csv", header=0)

# ����MSA����
genome_msa = GenomeMSA(msa_path)

# ����ģ��
model = AutoModel.from_pretrained(model_path)
model.eval()

# ��ʼ��һ��NumPy��������������
features_np = np.zeros((len(test_data), 128, 768), dtype=np.float32)

# ���������б���ȡ����
with torch.no_grad():  # ����Ҫ�����ݶȣ���ʡ�ڴ�ͼ�����Դ
    # ѭ��������������
    for i in range(len(test_data)):
        # �Ӳ��������л�ȡ��Ӧ����Ϣ
        chrom = test_data.iloc[i, 0]  # ��һ��
        start_pos = int(test_data.iloc[i, 1])  # �ڶ��е���ʼλ��
        end_pos = int(test_data.iloc[i, 1])  # �ڶ��еĽ���λ��

        # ��ȡ��ǵ�MSA����
        msa = genome_msa.get_msa(chrom, start_pos-64, end_pos+64, strand="+", tokenize=True)

        # ��MSA����ת��ΪTensor��ʽ
        msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64))

        # ����������������ֵ�����
        input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]

        # Ƕ���ʾ
        embedding = model(input_ids=input_ids, aux_features=aux_features).last_hidden_state

        # �������PyTorch����ת��ΪNumPy���鲢���浽features_np��
        features_np[i] = embedding.cpu().numpy()  # ȷ�������CPU�ϣ�Ȼ��ת��ΪNumPy����

        # Ƕ����ӻ�
#        embedding_df = pd.DataFrame(StandardScaler().fit_transform(embedding[0].numpy()))
#        embedding_df.index.name = "Position"
#        embedding_df.columns.name = "Embedding dimension"

# ��NumPy����ת��ΪPyTorch����������Ϊ.pth�ļ�
features_tensor = torch.from_numpy(features_np)
torch.save(features_tensor, '../data/test_GPN-MSA_feature.pth')