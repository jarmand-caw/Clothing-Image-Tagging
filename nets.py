import torch.nn as nn

class ColorNet(nn.Module): 
    def __init__(self):
        super(ColorNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,100)
        self.linear2 = nn.Linear(100,100)
        self.linear3 = nn.Linear(100,14)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear2(out)
        out = self.linear2(out)
        out = self.linear2(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out
        
class GenreNet(nn.Module): 
    def __init__(self):
        super(GenreNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,1000)
        self.linear2 = nn.Linear(1000,500)
        self.linear3 = nn.Linear(500,100)
        self.linear4 = nn.Linear(100,14)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer2(out)
        #out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out

class BodyNet(nn.Module): 
    def __init__(self):
        super(BodyNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,1000)
        self.linear2 = nn.Linear(1000,5)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer2(out)
        #out = self.layer2(out)
        #out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
        
class ChiefNet(nn.Module): 
    def __init__(self):
        super(ChiefNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,1000)
        self.linear2 = nn.Linear(1000,4)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer2(out)
        #out = self.layer2(out)
        #out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

class TypeNet(nn.Module): 
    def __init__(self):
        super(TypeNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,1000)
        self.linear2 = nn.Linear(1000,500)
        self.linear3 = nn.Linear(500,100)
        self.linear4 = nn.Linear(100,9)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer2(out)
        #out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        return out
        
class UseNet(nn.Module): 
    def __init__(self):
        super(UseNet, self).__init__()
        #feed forward layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(32*8*8,1000)
        self.linear2 = nn.Linear(1000,6)
        
        
        #activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #Use sigmoid to convert the output into range (0,1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer2(out)
        #out = self.layer2(out)
        #out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out