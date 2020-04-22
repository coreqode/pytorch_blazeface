from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch

class BlazeBlock(nn.Module):
    def __init__(self, inp, oup1, oup2=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_double_block = oup2 is not None
        self.use_pooling = self.stride != 1

        if self.use_double_block:
            self.channel_pad = oup2 - inp
        else:
            self.channel_pad = oup1 - inp

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=True),
            nn.BatchNorm2d(inp),
            # pw-linear
            nn.Conv2d(inp, oup1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup1),
        )
        self.act = nn.ReLU(inplace=True)

        if self.use_double_block:
            self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup1, oup1, kernel_size=kernel_size, stride=1, padding=padding, groups=oup1, bias=True),
                nn.BatchNorm2d(oup1),
                # pw-linear
                nn.Conv2d(oup1, oup2, 1, 1, 0, bias=True),
                nn.BatchNorm2d(oup2),
            )

        if self.use_pooling:
            self.mp = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)

    def forward(self, x):
        h = self.conv1(x)
        if self.use_double_block:
            h = self.conv2(h)

        # skip connection
        if self.use_pooling:
            x = self.mp(x)
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), 'constant', 0)
        return self.act(h + x)


def initialize(module):
    # original implementation is unknown
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


class BlazeFace(nn.Module):

    def __init__(self):
        super(BlazeFace, self).__init__()
        self.n_boxes = [2,6]
        self.n_classes = 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            BlazeBlock(48, 24, 96, stride=2),
            BlazeBlock(96, 24, 96),
            BlazeBlock(96, 24, 96),
        )
        self.fm = nn.Sequential(
            BlazeBlock(96, 24, 96, stride=2), #14 x 14
            BlazeBlock(96, 24, 96),
            BlazeBlock(96, 24, 96),
            )
        self.conf_14 = nn.Sequential(
            nn.Conv2d(96,self.n_boxes[0] * self.n_classes , kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.n_boxes[0] * self.n_classes),
            nn.Softmax(),
        )
        self.conf_7 = nn.Sequential(
            nn.Conv2d(96,self.n_boxes[1] * self.n_classes , kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.n_boxes[1] * self.n_classes),
            nn.Softmax(),
        )
        self.reg_14 = nn.Sequential(
            nn.Conv2d(96,self.n_boxes[0] * 4 , kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.n_boxes[0] * 4),
            nn.ReLU(inplace=True),
        )
        self.reg_7 = nn.Sequential(
            nn.Conv2d(96,self.n_boxes[1] * 4 , kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.n_boxes[1] * 4),
            nn.ReLU(inplace=True),
        )
        

        self.apply(initialize)

    def forward(self, x):
        
        batch_size = x.size(0)
        h = self.features(x)
        fm_14 = self.fm(h)
        fm_7 = self.fm(fm_14)
        conf_14 = self.conf_14(fm_14)
        reg_14 = self.reg_14(fm_14)
        conf_14_reshape = conf_14.view(batch_size, -1, 3)
        reg_14_reshape = reg_14.view(batch_size, -1,4)
        conf_7 = self.conf_7(fm_7)
        reg_7 = self.reg_7(fm_7)
        conf_7_reshape = conf_7.view(batch_size, -1, 3)
        reg_7_reshape = reg_7.view(batch_size, -1,4)

        conf_output = torch.cat([conf_7_reshape, conf_14_reshape], dim = 1)
        reg_output = torch.cat([reg_7_reshape, reg_14_reshape], dim = 1)

        return (conf_output, reg_output)
    
if __name__ == "__main__":
    model = BlazeFace()
    model = model.to('cuda')
    image = torch.randn(1, 3, 224, 224)
    image = image.to('cuda')
    output = model(image)
    print(output[0].size(), output[1].size())