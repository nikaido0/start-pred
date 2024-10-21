import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel import BaseModel


# 定义 DanQ 模型类，继承自 BaseModel
class DanQ(BaseModel):
    # 初始化方法，定义模型的超参数
    def __init__(
            self,
            filter=320,  # 卷积层的卷积核（滤波器）数量
            kernel=26,  # 卷积层的卷积核大小
            lstm_units=320,  # LSTM 层的单元数量
            dropout1=0.2,  # 第一个 Dropout 层的丢弃率
            dropout2=0.5,  # 第二个 Dropout 层的丢弃率
            dense=925,  # 全连接层的神经元数量
            classes=919,  # 输出的类别数
            shape=(1000, 4),  # 输入数据的形状（序列长度 1000，4 种碱基的 one-hot 编码）
            learning_rate=1e-3,  # 学习率
            pool=13,  # 最大池化层的池化窗口大小
            name='DanQ'  # 模型名称
    ):
        # 调用父类 BaseModel 的初始化方法
        super(DanQ, self).__init__(name=name, shape=shape, classes=classes, learning_rate=learning_rate)

        # 将参数存储为类的属性
        self.filter = filter
        self.kernel = kernel
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dense = dense
        self.lstm_units = lstm_units
        self.pool = pool

        # 定义模型结构
        self.conv = nn.Conv1d(in_channels=shape[1], out_channels=self.filter, kernel_size=self.kernel)  # 卷积层
        self.pool = nn.MaxPool1d(kernel_size=self.pool, stride=self.pool)  # 最大池化层
        self.lstm = nn.LSTM(input_size=self.filter, hidden_size=self.lstm_units, batch_first=True, bidirectional=True)  # 双向 LSTM
        self.dropout1_layer = nn.Dropout(self.dropout1)  # 第一个 Dropout 层
        self.dropout2_layer = nn.Dropout(self.dropout2)  # 第二个 Dropout 层
        self.fc1 = nn.Linear(2 * self.lstm_units, self.dense)  # 全连接层（由于是双向 LSTM，输出维度为 2 * lstm_units）
        self.fc2 = nn.Linear(self.dense, self.classes)  # 输出层

    # 前向传播函数
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, 1000, 4)

        # 卷积层
        x = F.relu(self.conv(x))
        x = self.pool(x)  # (batch_size, 320, 75)

        print(f"Shape after convolution and pooling: {x.shape}")

        # 确保输入 LSTM 的最后一个维度为 320
        if x.shape[1] != self.filter:  # 检查是否符合 LSTM 输入要求
            raise ValueError(f"Expected input size of {self.filter}, but got {x.shape[1]}.")

        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, 75, 320)

        # LSTM 层
        x, _ = self.lstm(x)

        # 取 LSTM 的最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, 2 * lstm_units)

        x = self.dropout1_layer(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2_layer(x)
        x = torch.sigmoid(self.fc2(x))

        return x


