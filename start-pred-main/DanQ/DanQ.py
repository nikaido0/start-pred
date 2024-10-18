# 导入 TensorFlow 和自定义的 BaseModel 类
import tensorflow as tf
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
        super(DanQ, self).__init__(
            name=name,
            shape=shape,
            classes=classes,
            learning_rate=learning_rate,
        )

        # 将参数存储为类的属性
        self.filter = filter
        self.kernel = kernel
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dense = dense
        self.lstm_units = lstm_units
        self.pool = pool

        # 设置随机种子，确保结果可复现
        tf.random.set_seed(0)

        # 调用构建模型的函数
        self.__build__()

    # 模型构建函数
    def __build__(
            self,
    ):
        # 定义输入层，形状为 (1000, 4)，即 DNA 序列长度为 1000，每个位置有 4 个可能的碱基（A、G、C、T）
        input = tf.keras.layers.Input(shape=self.shape)

        # 卷积层，使用 ReLU 激活函数，提取序列的局部特征
        conv = tf.keras.layers.Conv1D(
            self.filter,  # 卷积核的数量，即生成的特征图的数量
            kernel_size=self.kernel,  # 卷积核的大小
            strides=1,  # 步幅，表示卷积核滑动的步长
            padding='valid',  # 不使用填充，保持卷积的“有效”计算
            activation='relu'  # 使用 ReLU 激活函数
        )(input)

        # 最大池化层，减少特征图的大小，保留重要信息
        pool = tf.keras.layers.MaxPool1D(
            pool_size=self.pool,  # 池化窗口的大小
            strides=self.pool,  # 池化步幅
            padding='valid'  # 不进行填充
        )(conv)

        # 第一个 Dropout 层，随机丢弃部分神经元，防止过拟合
        drop1 = tf.keras.layers.Dropout(self.dropout1)(pool)

        # 双向 LSTM 层，用于提取序列的长程依赖特征
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=self.lstm_units)  # 每个方向的 LSTM 单元数量
        )(drop1)

        # 第二个 Dropout 层，进一步防止过拟合
        drop2 = tf.keras.layers.Dropout(self.dropout2)(lstm)

        # 展平层，将 LSTM 层的输出展平为一维向量，以便输入到全连接层
        flat = tf.keras.layers.Flatten()(drop2)

        # 全连接层，使用 ReLU 激活函数
        feed = tf.keras.layers.Dense(
            units=self.dense,  # 神经元的数量
            activation='relu'  # 使用 ReLU 激活函数
        )(flat)

        # 输出层，使用 sigmoid 激活函数，输出 919 个类别的预测值
        output = tf.keras.layers.Dense(
            units=self.classes,  # 输出的类别数量
            activation='sigmoid'  # 使用 sigmoid 激活函数，适合多标签分类
        )(feed)

        # 调用父类的方法，创建模型
        self.make_model(input, output)
