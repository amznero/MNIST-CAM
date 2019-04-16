import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, ):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # self.fc1 = nn.Linear(16, 10)
        # self.fc2 = nn.Linear(120, 84)
        self.classify = nn.Linear(16, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=8)
        # out = F.avg_pool2d(out, kernel_size=4)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))

        return self.classify(out)

if __name__ == '__main__':
    from torch.autograd import Variable
    model = LeNet()
    ipt = Variable(torch.randn(16, 1, 28, 28))
    out = model(ipt)
    print(out.size())
