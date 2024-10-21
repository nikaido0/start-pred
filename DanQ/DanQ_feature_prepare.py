# 导入所需的库
import argparse  # 用于解析命令行参数
import os
import sys

import h5py  # 用于创建和操作 HDF5 文件
import numpy as np  # 用于科学计算和处理数组
import torch  # 导入 PyTorch 库

sys.path.append('../start-pred-main/')  # 将上一级目录添加到 Python 路径中
from DanQ import DanQ  # 导入 DanQ 模型

SEQ_LENGTH = 1000  # DanQ 模型所接受的序列长度


# 对序列进行 one-hot 编码，将 DNA 序列转换为 DanQ 模型所需的输入格式
def one_hot(x):
    arr = np.array(list(x))  # 将输入序列转换为字符数组
    a = (arr == 'A')  # 如果字符是 A，则为 True，否则为 False
    g = (arr == 'G')  # 如果字符是 G，则为 True
    c = (arr == 'C')  # 如果字符是 C，则为 True
    t = (arr == 'T')  # 如果字符是 T，则为 True

    # 将四个布尔数组按顺序垂直堆叠，最后转置为 (4, 序列长度) 的矩阵
    return np.vstack([a, g, c, t]).T


# 读取并格式化序列数据
def get_sequence(filename):
    # 打开文件读取序列数据
    with open(filename, 'r') as f:
        seq = f.read()

    # 将序列按换行符分割为列表，去掉空行
    seq = seq.split('\n')
    seq = [s for s in seq if s]  # 去掉空行

    # 使用 one-hot 函数将序列进行编码
    seq = list(map(one_hot, seq))
    # 将编码后的序列转换为 numpy 数组，并将数据类型设置为 float32 以便 PyTorch 使用
    seq = np.array(seq).astype('float32')
    return seq


# 从 DanQ 模型中提取 919 个特征
def prepare_sequence(model, local_args, seq_type='ref'):
    # 获取参考序列或替代序列的文件名
    dataset_filename = os.path.join(local_args.path, f'test_{seq_type}-seq.fasta')

    # 读取并编码序列
    dataset_seq = get_sequence(dataset_filename)

    # 对序列进行截取，确保其长度符合 DanQ 模型的输入要求
    seq = np.array(dataset_seq)[:, :SEQ_LENGTH]

    # 将 numpy 数组转换为 PyTorch 张量
    seq_tensor = torch.tensor(seq)

    # 使用 DanQ 模型预测该序列的特征，返回特征向量
    return model.predict(seq_tensor)


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

    # 初始化 DanQ 模型
    danQ = DanQ(name='DanQ')

    print('Process ref:')
    # 处理参考序列，并提取其特征
    ref = prepare_sequence(danQ, args, seq_type='ref')

    print('Process alt:')
    # 处理替代序列，并提取其特征
    alt = prepare_sequence(danQ, args, seq_type='alt')

    # 将提取的特征保存到一个 HDF5 文件中
    file_path = os.path.join(args.path, '../data/test_DanQ_feature.h5')  # 输出文件路径
    hf = h5py.File(file_path, 'w')  # 打开 HDF5 文件，准备写入
    hf.create_dataset('feat_ref', data=ref.numpy())  # 创建数据集存储参考序列的特征，并转换为 numpy 数组
    hf.create_dataset('feat_alt', data=alt.numpy())  # 创建数据集存储替代序列的特征，并转换为 numpy 数组
    hf.close()  # 关闭 HDF5 文件
