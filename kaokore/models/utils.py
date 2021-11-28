from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()
        mid_channels = mid_channels or out_channels
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        ]
        if bn: layers.insert(2, nn.BatchNorm2d(mid_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

def add_res_block(layers, in_channels, out_channels, mid_channels=None, bn=False, final_bn=False):
    layers.append(ResBlock(in_channels, out_channels, mid_channels, bn))
    if final_bn: layers.append(nn.BatchNorm2d(out_channels))

def add_conv2d_block(layers, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   activation=nn.ReLU(inplace=True), bn=False):
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)

def add_convtranspose2d_block(layers, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   activation=nn.ReLU(inplace=True), bn=False):
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
    if bn: layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation=nn.ReLU(inplace=True), bn=False):
        super(Conv2dBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        ]
        if bn: layers.insert(1, nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation=nn.ReLU(inplace=True), bn=False):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            activation
        ]
        if bn: layers.insert(1, nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

