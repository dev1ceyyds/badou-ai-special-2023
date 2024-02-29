import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    # 参数有：net网络模型，cost损失函数，optimist优化函数
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    #     定义cost函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    #     定义优化函数

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    # 定义训练 train_loader训练数据，epoches训练代数为3（epoches代，表示train_loader数据集全部训练一次）
    def train(self, train_loader, epoches=3):
        # 三代循环
        for epoch in range(epoches):
            running_loss = 0.0
            # 所有数据循环
            for i, data in enumerate(train_loader, 0):
                # data中包含输入矩阵和labless矩阵
                inputs, labels = data
                # 优化初始
                self.optimizer.zero_grad()
                # 正向+反向+优化
                # 正向
                outputs = self.net(inputs)
                # 计算损失
                loss = self.cost(outputs, labels)
                # 反向传播
                loss.backward()
                # 优化
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[代数： %d, %.2f%%] loss损失: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('训练结束')

    # 推理 test_loader推理数据
    def evaluate(self, test_loader):
        print("开始推理")
        correct = 0
        total = 0
        with torch.no_grad():  # 当执行模型以获取预测结果而不是为了反向传播或微调权重时，可以避免不必要的计算开销，从而提高执行效率。
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                # 用argmax函数找到最大概率值的角标 1代表只找1个
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('准确性: %d %%' % (100 * correct / total))

# 加载数据集
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 构建输入层、隐藏层（两个隐藏层）、输出层
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    # 神经网络一次正向
    def forward(self, x):
        # 把输入的x矩阵化
        x = x.view(-1, 28 * 28)
        # 全连接结果过激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x





if __name__ == '__main__':
    # 构建神经网络
    net = MnistNet()
    # 构建完整的模型
    model = Model(net, "MSE", "RMSP")
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
