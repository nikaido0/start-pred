import torch
import torch.nn as nn
import torch.optim as optim
import os

# 通用的框架，用于构建各种深度学习模型
class BaseModel(nn.Module):
    def __init__(
            self,
            classes=919,  # 输出类别数，默认是919
            shape=(1000, 4),  # 输入数据的形状，默认是1000个时间步，4个特征
            learning_rate=1e-3,  # 学习率，默认是0.001
            name='BaseModel',  # 模型名称，默认是BaseModel
            checkpoint_dir='../checkpoint/'  # 模型权重保存路径
    ):
        super(BaseModel, self).__init__()

        # 模型权重保存路径，按模型名称区分
        self.checkpoint = os.path.join(checkpoint_dir, name)
        self.classes = classes  # 类别数
        self.shape = shape  # 输入数据形状
        self.learning_rate = learning_rate  # 学习率
        self.name = name  # 模型名称
        self.model = None  # PyTorch模型对象，初始化为空
        self.optimizer = None  # 优化器，初始化为空
        self.criterion = nn.BCEWithLogitsLoss()  # 默认损失函数

    def __build__(self):
        """子类需要实现这个方法，定义具体的模型架构。"""
        raise NotImplementedError("Please implement the __build__ method in the child class.")

    def scheduler(self, epoch):
        """学习率调度器，基于训练轮次动态调整学习率。"""
        decay_start_epoch = 5
        decay_factor = 0.1

        if epoch < decay_start_epoch:
            return self.learning_rate  # 保持学习率不变
        else:
            # 使用torch.tensor将decay_factor转换为张量
            return self.learning_rate * torch.exp(-torch.tensor(decay_factor))  # 指数衰减

    def fit(self, data_loader, epochs=200, validation_loader=None, early_stop_patience=5):
        """
        训练模型，使用早停、学习率调度和检查点保存。

        参数:
        - data_loader: 训练数据集的迭代器
        - epochs: 训练的轮数，默认200
        - validation_loader: 验证集数据，默认无
        - early_stop_patience: 早停法的容忍度，默认为5
        """
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.train()  # 设置模型为训练模式
            running_loss = 0.0

            for inputs, targets in data_loader:  # 训练数据的迭代器
                self.optimizer.zero_grad()  # 清零梯度
                outputs = self(inputs)  # 前向传播
                loss = self.criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新权重

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}')

            # 学习率调度
            self.learning_rate = self.scheduler(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            # 验证
            if validation_loader:
                val_loss = self.validate(validation_loader)
                # 早停机制
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self.save_model()  # 保存最佳模型
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print("Early stopping triggered.")
                        break

    def validate(self, data_loader):
        self.eval()  # 设置模型为评估模式
        total_loss = 0.0

        with torch.no_grad():  # 不计算梯度
            for inputs, targets in data_loader:
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

    def predict(self, x):
        """
        使用训练好的模型进行预测。

        参数:
        - x: 输入数据
        """
        self.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度
            return self(x)

    def save_model(self):
        """
        保存整个模型（结构+权重）。
        """
        torch.save(self.state_dict(), self.checkpoint)
        print(f'Model weights saved at {self.checkpoint}')

    def load_weights(self):
        """加载保存的权重，如果存在则加载。"""
        if os.path.exists(self.checkpoint):
            self.load_state_dict(torch.load(self.checkpoint))
            print(f"Weights loaded from {self.checkpoint}")
        else:
            print(f"No weights found at {self.checkpoint}. Starting from scratch.")

    def make_model(self):
        """
        构建并编译模型，允许子类定义具体的网络结构。

        参数:
        - input_shape: 输入层的形状
        - output_shape: 输出层的形状
        """
        self.model = self.__build__()  # 创建模型
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # 初始化优化器
        self.load_weights()  # 加载权重
