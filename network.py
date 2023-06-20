import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, s_channel, v_channel):
        super(Network, self).__init__()
        self.conv_in = nn.Sequential(conv2d(3, 8, 3), conv2d(8, 8, 3), conv2d(8, 16, 3), conv2d(16, 16, 3))
        self.conv_s = nn.Sequential(conv2d(16, 16, 3), conv2d(16, s_channel, 3, activation=nn.Sigmoid()))
        self.activation_s = nn.ReLU()
        self.conv_v = nn.Sequential(conv2d(16, 16, 3), conv2d(16, 16, 3, 2), conv2d(16, 16, 3, 3), conv2d(16, v_channel, 3),
                                    conv2d(v_channel, v_channel, 3, 2), conv2d(v_channel, v_channel, 3, 3, activation=nn.Softmax(dim=1)))
        self.conv_out = nn.Sequential(conv2d(v_channel, 8, 3), conv2d(8, 8, 3), conv2d(8, 4, 3), conv2d(4, 4, 3), conv2d(4, 3, 1, activation=nn.Sigmoid()))
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def cos_dis(self, x, y):
        return 1 - self.cos_sim(x, y)

    def forward(self, x):
        x_in = self.conv_in(x)

        # x_s
        f_s = self.conv_s(x_in)
        x_s = torch.nn.functional.adaptive_avg_pool2d(f_s, (1, 1))

        # x_v
        x_v = self.conv_v(x_in)

        x_fushion = x_s * x_v

        x_out = self.conv_out(x_fushion)
        return x_s, x_v, x_out

    def transfer_full(self, x, y):
        x_s, x_v, x_out = self.forward(x)
        y_s, y_v, y_out = self.forward(y)

        x2y_fushion = y_s * x_v
        x2y_out = self.conv_out(x2y_fushion)

        y2x_fushion = x_s * y_v
        y2x_out = self.conv_out(y2x_fushion)

        return x_s, x_v, x_out, y_s, y_v, y_out, x2y_out, y2x_out
    
    def transfer_x2y(self, x, y):
        x_v = self.conv_v(self.conv_in(x))
        y_s = self.conv_s(self.conv_in(y))
        y_s = torch.nn.functional.adaptive_avg_pool2d(y_s, (1, 1))

        x2y_fushion = y_s * x_v
        x2y_out = self.conv_out(x2y_fushion)

        return x2y_out


    def transfer_resize(self, x, y):
        x_s, x_v, x_out = self.forward(x)
        y_s, y_v, y_out = self.forward(y)
        x_resize = nn.functional.interpolate(x, (256, 256), mode='bicubic', align_corners=False)
        y_resize = nn.functional.interpolate(y, (256, 256), mode='bicubic', align_corners=False)
        x_s, _, _ = self.forward(x_resize)
        y_s, _, _ = self.forward(y_resize)

        x2y_fushion = y_s * x_v
        x2y_out = self.conv_out(x2y_fushion)

        y2x_fushion = x_s * y_v
        y2x_out = self.conv_out(y2x_fushion)

        return x_s, x_v, x_out, y_s, y_v, y_out, x2y_out, y2x_out

    def get_v(self, x):
        x_in = self.conv_in(x)
        x_v = self.conv_v(x_in)
        return x_v

    def get_s(self, x):
        x_resize = nn.functional.interpolate(x, (256, 256), mode='bicubic', align_corners=False)
        x_in = self.conv_in(x_resize)
        f_s = self.conv_s(x_in)
        x_s = torch.nn.functional.adaptive_avg_pool2d(f_s, (1, 1))
        return x_s

    def fushion_sv(self, x_s, x_v):
        x_fushion = x_s * x_v
        x_out = self.conv_out(x_fushion)
        return x_out


def conv2d(in_channels, out_channels, kernel_size, dilation_rate=1, activation=nn.ReLU()):
    padding = dilation_rate
    if kernel_size == 1:
        padding = 0
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation_rate, padding=padding), activation)
