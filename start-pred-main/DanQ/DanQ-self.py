import tensorflow as tf


# 定义 DanQ 模型
class DanQ(tf.keras.Model):
    def __init__(self, seq_length=1000, filter_size=320, kernel_size=26, lstm_units=320, pool_size=13, dense_units=925,
                 output_units=919, dropout1=0.2, dropout2=0.5):
        super(DanQ, self).__init__()

        # 定义卷积层
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filter_size,  # 卷积核的数量
            kernel_size=kernel_size,  # 卷积核的大小
            activation='relu',  # 使用 ReLU 激活函数
            padding='valid'  # 不使用填充
        )

        # 定义最大池化层
        self.max_pooling = tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size)

        # 定义第一个 Dropout 层
        self.dropout1 = tf.keras.layers.Dropout(dropout1)

        # 定义双向 LSTM 层
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units))

        # 定义第二个 Dropout 层
        self.dropout2 = tf.keras.layers.Dropout(dropout2)

        # 定义全连接层
        self.dense1 = tf.keras.layers.Dense(units=dense_units, activation='relu')

        # 定义输出层
        self.output_layer = tf.keras.layers.Dense(units=output_units, activation='sigmoid')

    def call(self, inputs):
        # 前向传播：从输入到输出的计算过程
        x = self.conv1d(inputs)  # 通过卷积层
        x = self.max_pooling(x)  # 通过最大池化层
        x = self.dropout1(x)  # 应用第一个 Dropout 层
        x = self.bi_lstm(x)  # 通过双向 LSTM 层
        x = self.dropout2(x)  # 应用第二个 Dropout 层
        x = self.dense1(x)  # 通过全连接层
        return self.output_layer(x)  # 输出预测结果


# 测试模型定义
if __name__ == "__main__":
    # 假设输入是 (batch_size, sequence_length, num_features)
    model = DanQ(seq_length=1000)
    inputs = tf.random.normal([32, 1000, 4])  # 32个长度为1000的DNA序列，4个可能的碱基（A, T, C, G）
    outputs = model(inputs)
    print(outputs.shape)  # 输出形状应为 (32, 919)，对应919个预测结果
