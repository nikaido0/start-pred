#!/usr/bin/env python
# coding=gbk

# 导入所需的库
import os
import h5py  # 用于创建和操作HDF5文件
import numpy as np  # 用于科学计算和处理数组
import pandas as pd  # 用于数据处理
import argparse  # 用于解析命令行参数

import sys

sys.path.append('../')  # 将上一级目录添加到Python路径中
from DanQ import DanQ  # 导入DanQ模型

SEQ_LENGTH = 1000  # DanQ模型所接受的序列长度


# 对序列进行one-hot编码，将DNA序列转换为DanQ模型所需的输入格式
def one_hot(x):
    arr = np.array(list(x))  # 将输入序列转换为字符数组
    a = (arr == 'A')  # 如果字符是A，则为True，否则为False
    g = (arr == 'G')  # 如果字符是G，则为True
    c = (arr == 'C')  # 如果字符是C，则为True
    t = (arr == 'T')  # 如果字符是T，则为True

    # 将四个布尔数组按顺序垂直堆叠，最后转置为(4, 序列长度)的矩阵
    return np.vstack([a, g, c, t]).T


# 读取并格式化序列数据
def get_sequence(filename):
    # 打开文件读取序列数据
    with open(filename, 'r') as f:
        seq = f.read()

    # 将序列按换行符分割为列表，去掉空行
    seq = seq.split('\n')[0:]
    # 使用one-hot函数将序列进行编码
    seq = list(map(one_hot, seq))
    # 将编码后的序列转换为numpy数组，并将数据类型设置为uint8
    seq = np.array(seq).astype('uint8')
    return seq


# 从DanQ模型中提取919个特征
def prepare_sequence(model, args, type='ref'):
    # 获取参考序列或替代序列的文件名
    dataset_filename = os.path.join(args.path, f'test_{type}-seq.fasta')

    # 读取并编码序列
    dataset_seq = get_sequence(dataset_filename)

    # 对序列进行截取，确保其长度符合DanQ模型的输入要求
    seq = np.array(dataset_seq)[:, :SEQ_LENGTH]

    # 使用DanQ模型预测该序列的特征，返回特征向量
    return model.predict(seq)


# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Arguments for preparing features.'  # 解析准备特征所需的参数
    )
    # 添加参数 --path，表示输入数据的路径
    parser.add_argument(
        '--path', default='./', type=str, help='path for input data'
    )
    args = parser.parse_args()  # 解析命令行参数

    # 初始化DanQ模型
    danQ = DanQ(name='DanQ')

    print('Process ref:')
    # 处理参考序列，并提取其特征
    ref = prepare_sequence(danQ, args, type='ref')

    print('Process alt:')
    # 处理替代序列，并提取其特征
    alt = prepare_sequence(danQ, args, type='alt')

    # 将提取的特征保存到一个HDF5文件中
    file_path = os.path.join(args.path, '../data/test_DanQ_feature.h5')  # 输出文件路径
    hf = h5py.File(file_path, 'w')  # 打开HDF5文件，准备写入
    hf.create_dataset('feat_ref', data=ref)  # 创建数据集存储参考序列的特征
    hf.create_dataset('feat_alt', data=alt)  # 创建数据集存储替代序列的特征
    hf.close()  # 关闭HDF5文件
