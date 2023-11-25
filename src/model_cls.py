from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.down = nn.MaxPool2d(2, 2) 
        self.up = nn.Upsample(size=(400, 400), mode='bilinear')

    def forward(self, x):
        x = self.conv1(x) # output: 32 x 396 x 396
        x = self.relu(x)
        x = self.down(x) # 32 x 198 x 198
        x = self.conv2(x) # 64 x 194 x 194
        x = self.relu(x)
        x = self.down(x) # 64 x 97 x 97
        x = self.conv3(x) # 2 x 97 x 97
        x = self.sigmoid(x)
        x = self.up(x) # 1 x 400 x 400
        return x