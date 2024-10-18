import tensorflow as tf

###
class BaseModel:
    def __init__(
            self,
            classes=919,  # 模型的输出类别数，默认值为919
            shape=(1000, 4),  # 输入数据的形状，默认为1000个时间步，4个特征（如DNA序列的A、T、C、G）
            learning_rate=1e-3,  # 学习率，控制模型训练时的步长
            name='BaseModel'  # 模型的名称，默认值为'BaseModel'
    ):
        # 模型的checkpoint路径，保存和加载模型的权重
        self.checkpoint = f'../checkpoint/{name}'
        self.classes = classes  # 类别数
        self.shape = shape  # 输入形状
        self.learning_rate = learning_rate  # 学习率
        self.name = name  # 模型的名称
        self.model = None  # Keras模型实例，初始化时为空

    def __build__(
            self,
    ):
        # 留给子类（如DanQ）实现的模型结构构建方法
        pass

    def scheduler(
            self,
            epoch,
            lr
    ):
        # 学习率调度器，动态调整学习率
        # 如果当前epoch小于5，保持原来的学习率
        # 否则，按照指数衰减的方式减少学习率
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    def fit(
            self,
            dataiter,  # 训练数据的迭代器
            epochs=200,  # 训练的轮数
            validation_data=None  # 验证集数据，用于评估模型的性能
    ):
        # 学习率调度回调，逐步减少学习率
        learning_rate = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        # 早停法，监控验证集损失，如果损失不再下降，提前停止训练
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # 监控验证集的损失
            patience=5,  # 如果验证损失在5个epoch内不再改善，则停止训练
            mode='auto',  # 自动选择最合适的监控模式
            restore_best_weights=True  # 恢复到验证集上表现最好的模型权重
        )
        # 模型检查点回调，每个epoch保存模型权重
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint,  # 保存的路径
            monitor='val_loss',  # 监控验证集的损失
            verbose=0,  # 日志等级
            save_weights_only=True,  # 只保存模型的权重
            mode='auto',  # 自动选择保存的模式
            save_freq='epoch'  # 每个epoch保存一次
        )
        # 开始训练模型
        self.model.fit(
            dataiter,  # 训练数据
            epochs=epochs,  # 总训练轮数
            callbacks=[early_stop, learning_rate, checkpoint],  # 回调函数：学习率调度、早停、权重保存
            validation_data=validation_data,  # 验证数据
            verbose=1  # 显示详细训练过程
        )

    def predict(
            self,
            x  # 输入数据
    ):
        # 使用训练好的模型进行预测
        return self.model.predict(x)

    def save_model(
            self,
            model_name  # 保存模型的路径
    ):
        # 保存整个模型，包括结构和权重
        self.model.save(model_name)

    def make_model(
            self,
            input,  # 模型的输入层
            output,  # 模型的输出层
            loss='binary_crossentropy'  # 损失函数，默认是二元交叉熵，用于二分类或多标签分类
    ):
        # 构建模型，将输入层和输出层连接起来
        self.model = tf.keras.models.Model(
            inputs=input, outputs=output, name=self.name
        )
        # 使用Adam优化器，并设置学习率
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # 编译模型，定义损失函数和评估指标
        self.model.compile(
            optimizer=opt,
            loss=loss,  # 损失函数
            metrics=[tf.keras.metrics.AUC(name='auc')]  # 使用AUC作为评估指标
        )

        # 尝试加载模型的权重（如果存在之前保存的权重）
        try:
            self.model.load_weights(self.checkpoint)
            print('weights loaded from', self.checkpoint)  # 成功加载权重
        except Exception as e:
            print('no weights at', self.checkpoint)  # 没有找到保存的权重
            print(e)
