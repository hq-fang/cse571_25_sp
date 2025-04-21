import torch.nn as nn

class ObservationModel(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.conv_stack = None 
        self.mlp = None
        # YOUR IMPLEMENTATION HERE
        # you are expected to implement a 4 layer CNN followed by 2 fully connected layers
        # self.conv_stack should be a nn.Sequential object containing the following layers:
        # 1. Conv2d with suitable input channels (figure it out yourself!), 32 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 2. Conv2d with 32 input channels, 64 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 3. Conv2d with 64 input channels, 128 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 4. Conv2d with 128 input channels, 256 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3,    32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,   64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # self.mlp should be a nn.Sequential object containing the following layers:
        # 1. Linear layer with suitable input channels (figure it out yourself!), 512 output channels, ReLU
        # 2. Linear layer with 512 input channels, 512 output channels, ReLU
        # 3. Linear layer with 512 input channels, suitable output features (figure it out yourself!)

        flatten_size = 256 * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )

        # These functions may be useful: nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear
        
        # YOUR IMPLEMENTATION END HERE

    def forward(self, x):
        x = self.conv_stack(x)
        b,c,h,w = x.shape
        x = x.reshape(b,-1)
        x = self.mlp(x)
        return x



class ObservationModelExtra(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.conv_stack = None 
        self.mlp = None
        # YOUR IMPLEMENTATION HERE
        # you are expected to implement a 4 layer CNN followed by 2 fully connected layers
        # self.conv_stack should be a nn.Sequential object containing the following layers:
        # 1. Conv2d with suitable input channels (figure it out yourself!), 32 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 2. Conv2d with 32 input channels, 64 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 3. Conv2d with 64 input channels, 128 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2
        # 4. Conv2d with 128 input channels, 256 output channels, kernel_size=3, padding=1, ReLU, MaxPool2d with kernel_size=2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3,    32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,   64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # self.mlp should be a nn.Sequential object containing the following layers:
        # 1. Linear layer with suitable input channels (figure it out yourself!), 512 output channels, ReLU
        # 2. Linear layer with 512 input channels, 512 output channels, ReLU
        # 3. Linear layer with 512 input channels, suitable output features (figure it out yourself!)

        flatten_size = 256 * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_channels),
        )

        # These functions may be useful: nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear
        
        # YOUR IMPLEMENTATION END HERE

    def forward(self, x):
        x = self.conv_stack(x)
        b,c,h,w = x.shape
        x = x.reshape(b,-1)
        x = self.mlp(x)
        return x
