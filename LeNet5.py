from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
class LeNet_5(nn.Module): #inheritance
    def __init__(self):
        super(LeNet_5,self).__init__()#부모클래스를 초기화 해줌으로써 여기서도 멤버를 사용할 수있음 (nn. 이런게 다 사용하는 거)
        #super().__init__()도 가능
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)#in_channel(RGB:3), out_channel(numberofkernel)
        self.conv2 = nn.Conv2d(6,16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16,120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84,10)
    def forward(self, x):
        x = F.tanh(self.conv1(x))#tanh: activation
        x = F.avg_pool2d(x,2,2)#input , kernel size, stride
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x,2,2)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)#view = reshape
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if __name__=="__main__":
    model = LeNet_5()
    model.to(torch.device('cpu'))
    summary(model, input_size=(1,32,32))
