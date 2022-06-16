import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# best net
class MNlexNet(nn.Module):
    def __init__(self):
        super(MNlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1), # 28*28*20
            nn.MaxPool2d(2,2), # 14*14*20
            nn.ReLU(inplace=True),
            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, padding=1), # 14*14*40
            nn.MaxPool2d(2,2), # 7*7*40
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(40, 100, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 40, kernel_size=3, padding=1),
            nn.MaxPool2d(3,2), # 3*3*40
            nn.ReLU(inplace=True),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(40*3*3, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 10),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        in_size = x.size(0)
        x = self.layer1(x) # 256*20*14*14
        x = self.layer2(x) # 256*40*7*7
        x = self.layer3(x) # 256*40*3*3
        x = x.view(in_size,-1) # 256*360
        x = self.fc(x) # 256*10
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4,input_shape=(28, 28, 1)), 
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output