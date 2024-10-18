import torch
import torch.nn as nn
import torch.optim as optim
import os


class BaseModel(nn.Module):
    def __init__(
            self,
            classes=919,  # 模型的输出类别数，默认值为919
            shape=(1000, 4),  # 输入数据的形状，默认为1000个时间步，4个特征
            learning_rate=1e-3,  # 学习率，控制模型训练时的步长
            name='BaseModel'  # 模型的名称，默认值为'BaseModel'
    ):
        super(BaseModel, self).__init__()

        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # 输出使用的设备信息

        # 模型的checkpoint路径，保存和加载模型的权重
        self.checkpoint = f'../checkpoint/{name}'  # 模型权重保存路径
        self.classes = classes  # 类别数
        self.shape = shape  # 输入形状
        self.learning_rate = learning_rate  # 学习率
        self.name = name  # 模型的名称
        self.model = None  # PyTorch模型实例，初始化时为空
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # Adam优化器
        self.criterion = nn.BCEWithLogitsLoss()  # 默认损失函数，适用于二分类和多标签分类

        # 将模型迁移到指定设备（CPU或GPU）
        self.to(self.device)

    def __build__(self, input_shape, output_shape):
        """
        留给子类实现的模型结构构建方法。
        子类需要实现这个方法来定义具体的网络架构。
        """
        raise NotImplementedError("Please implement the __build__ method in the child class.")

    def scheduler(self, epoch):
        """
        学习率调度器，动态调整学习率。
        - 在前5个epoch保持原学习率
        - 之后按指数衰减的方式减少学习率
        """
        if epoch < 5:
            return self.learning_rate  # 训练初期学习率不变
        else:
            # 照指数衰减的方式减少学习率
            return self.learning_rate * torch.exp(torch.tensor(-0.1))

    def fit(self, data_loader, epochs=200, validation_loader=None):
        """
        训练模型，使用早停、学习率调度和检查点保存。
        参数:
        - data_loader: 训练数据的迭代器
        - epochs: 训练的轮数，默认200
        - validation_loader: 验证集数据，默认无
        """
        best_loss = float('inf')  # 初始化最佳损失为无穷大
        patience_counter = 0  # 早停容忍度计数器

        for epoch in range(epochs):
            self.train()  # 设置模型为训练模式
            running_loss = 0.0  # 初始化当前epoch的损失

            for inputs, targets in data_loader:  # 训练数据的迭代器
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 将输入和目标迁移到GPU
                self.optimizer.zero_grad()  # 清零梯度
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新权重

                running_loss += loss.item()  # 累加当前batch的损失

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}')

            # 学习率调度
            self.learning_rate = self.scheduler(epoch)  # 获取新学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate  # 更新优化器的学习率

            # 验证
            if validation_loader:
                val_loss = self.validate(validation_loader)  # 验证集上的损失
                if val_loss < best_loss:
                    best_loss = val_loss  # 更新最佳损失
                    patience_counter = 0  # 重置耐心计数器
                    self.save_model()  # 保存最佳模型
                else:
                    patience_counter += 1  # 如果没有改进，则增加耐心计数
                    if patience_counter >= 5:  # 早停容忍度
                        print("Early stopping triggered.")  # 触发早停
                        break

    def validate(self, data_loader):
        """
        验证模型性能，计算验证集上的损失。
        参数:
        - data_loader: 验证数据的迭代器
        返回:
        - avg_loss: 验证集上的平均损失
        """
        self.eval()  # 设置模型为评估模式
        total_loss = 0.0  # 初始化总损失
        with torch.no_grad():  # 不计算梯度
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 将输入和目标迁移到GPU
                outputs = self.model(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                total_loss += loss.item()  # 累加损失

        avg_loss = total_loss / len(data_loader)  # 计算平均损失
        print(f'Validation Loss: {avg_loss:.4f}')  # 输出验证损失
        return avg_loss  # 返回平均损失

    def predict(self, x):
        """
        使用训练好的模型进行预测。
        参数:
        - x: 输入数据
        返回:
        - 模型的预测结果
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度
            x = x.to(self.device)  # 将输入迁移到GPU
            return self.model(x)  # 返回预测结果

    def save_model(self):
        """
        保存整个模型（结构+权重）。
        """
        torch.save(self.state_dict(), self.checkpoint)  # 保存模型状态字典
        print(f'Model weights saved at {self.checkpoint}')  # 输出保存路径

    def load_weights(self):
        """
        加载保存的权重，如果存在则加载。
        """
        if os.path.exists(self.checkpoint):  # 检查权重文件是否存在
            self.load_state_dict(torch.load(self.checkpoint))  # 加载模型权重
            print(f"Weights loaded from {self.checkpoint}")  # 输出加载路径
        else:
            print(f"No weights found at {self.checkpoint}. Starting from scratch.")  # 输出未找到权重

    def make_model(self, input_shape, output_shape):
        """
        构建并编译模型，允许子类定义具体的网络结构。

        参数:
        - input_shape: 输入层的形状
        - output_shape: 输出层的形状
        """
        self.model = self.__build__(input_shape, output_shape)  # 创建模型
        self.load_weights()  # 尝试加载权重
