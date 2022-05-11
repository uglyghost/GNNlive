import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature, n_hidden, n_output):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(Net, self).__init__()
        # 此步骤是官方要求
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 设置输入层到隐藏层的函数
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.hidden(x))
        # 给x加权成为a，用激励函数将a变成特征b
        x = self.predict(x)
        # 给b加权，预测最终结果
        return x