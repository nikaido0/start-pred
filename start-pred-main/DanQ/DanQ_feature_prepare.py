#!/usr/bin/env python
# coding=gbk

# ��������Ŀ�
import os
import h5py  # ���ڴ����Ͳ���HDF5�ļ�
import numpy as np  # ���ڿ�ѧ����ʹ�������
import pandas as pd  # �������ݴ���
import argparse  # ���ڽ��������в���

import sys

sys.path.append('../')  # ����һ��Ŀ¼��ӵ�Python·����
from DanQ import DanQ  # ����DanQģ��

SEQ_LENGTH = 1000  # DanQģ�������ܵ����г���


# �����н���one-hot���룬��DNA����ת��ΪDanQģ������������ʽ
def one_hot(x):
    arr = np.array(list(x))  # ����������ת��Ϊ�ַ�����
    a = (arr == 'A')  # ����ַ���A����ΪTrue������ΪFalse
    g = (arr == 'G')  # ����ַ���G����ΪTrue
    c = (arr == 'C')  # ����ַ���C����ΪTrue
    t = (arr == 'T')  # ����ַ���T����ΪTrue

    # ���ĸ��������鰴˳��ֱ�ѵ������ת��Ϊ(4, ���г���)�ľ���
    return np.vstack([a, g, c, t]).T


# ��ȡ����ʽ����������
def get_sequence(filename):
    # ���ļ���ȡ��������
    with open(filename, 'r') as f:
        seq = f.read()

    # �����а����з��ָ�Ϊ�б�ȥ������
    seq = seq.split('\n')[0:]
    # ʹ��one-hot���������н��б���
    seq = list(map(one_hot, seq))
    # ������������ת��Ϊnumpy���飬����������������Ϊuint8
    seq = np.array(seq).astype('uint8')
    return seq


# ��DanQģ������ȡ919������
def prepare_sequence(model, args, type='ref'):
    # ��ȡ�ο����л�������е��ļ���
    dataset_filename = os.path.join(args.path, f'test_{type}-seq.fasta')

    # ��ȡ����������
    dataset_seq = get_sequence(dataset_filename)

    # �����н��н�ȡ��ȷ���䳤�ȷ���DanQģ�͵�����Ҫ��
    seq = np.array(dataset_seq)[:, :SEQ_LENGTH]

    # ʹ��DanQģ��Ԥ������е�������������������
    return model.predict(seq)


# ���������
if __name__ == '__main__':
    # ���������в���������
    parser = argparse.ArgumentParser(
        description='Arguments for preparing features.'  # ����׼����������Ĳ���
    )
    # ��Ӳ��� --path����ʾ�������ݵ�·��
    parser.add_argument(
        '--path', default='./', type=str, help='path for input data'
    )
    args = parser.parse_args()  # ���������в���

    # ��ʼ��DanQģ��
    danQ = DanQ(name='DanQ')

    print('Process ref:')
    # ����ο����У�����ȡ������
    ref = prepare_sequence(danQ, args, type='ref')

    print('Process alt:')
    # ����������У�����ȡ������
    alt = prepare_sequence(danQ, args, type='alt')

    # ����ȡ���������浽һ��HDF5�ļ���
    file_path = os.path.join(args.path, '../data/test_DanQ_feature.h5')  # ����ļ�·��
    hf = h5py.File(file_path, 'w')  # ��HDF5�ļ���׼��д��
    hf.create_dataset('feat_ref', data=ref)  # �������ݼ��洢�ο����е�����
    hf.create_dataset('feat_alt', data=alt)  # �������ݼ��洢������е�����
    hf.close()  # �ر�HDF5�ļ�
