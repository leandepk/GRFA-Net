import torch
import torch.nn as nn
import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=3,dilation=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

# 定义共享分离卷积
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)

# 定义共享分离卷积完成


# 定义通道混洗层
def channel_shuffle(x, groups):
    # print(x.shape)
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
# 定义通道混洗层

# 定义注意力机制
# class Dw_Attention(nn.Module):
#     def __init__(self, in_planes):
#         super(Dw_Attention, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes,in_planes,3,1,1,groups=in_planes,bias=False) # dw_conv
#         self.relu1 = nn.ReLU()
#         self.conv2 = conv1x1(in_planes,in_planes)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         out = self.relu1(self.conv1(x))
#         out = self.conv2(out)
#         return self.sigmoid(out)

# class Pw_Attention(nn.Module):
#     def __init__(self, in_planes):
#         super(Pw_Attention, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes,in_planes,1,bias=False) # pw_conv
#         self.relu1 = nn.ReLU()
#         self.conv2 = conv1x1(in_planes,in_planes)
#         self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     out = self.relu1(self.conv1(x))
    #     out = self.conv2(out)
    #     return self.sigmoid(out)

# 定义第二种注意力机制CP
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel//2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//2, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
# 第二种注意力机制
# 第三种注意力机制

class DP_attention(nn.Module):
    def __init__(self, in_planes):
        super(DP_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,in_planes,3,1,1,groups=in_planes,bias=False) # dw_conv
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)  # pw_conv
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        return self.sigmoid(out)

# 第三种注意力力机制模块定义

# 定义深度分离卷积
class DP_layer(nn.Module):
    def __init__(self,in_planes):
        super(DP_layer,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1, groups=in_planes, bias=False)  # dw_conv
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)  # pw_conv
        self.relu2 = nn.ReLU()

    def forward(self,x):
        y1 = self.relu1(self.conv1(x))
        y2 = self.relu2(self.conv2(y1))
        return y2


# 定义基本block
class Block(nn.Module):
    def __init__(self,in_planes):
        super(Block,self).__init__()
        self.conv1 = conv3x3(in_planes,in_planes)
        self.in1 = nn.InstanceNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes*2,in_planes,1,bias=True) # 减少通道操作
        self.dp_attention = DP_attention(in_planes)
        self.ca_attention = CALayer(in_planes)
        self.pa_attention = PALayer(in_planes)

    def forward(self,x):
        out = self.relu1(self.in1(self.conv1(x)))
        out = torch.cat((x, out), 1)
        out = channel_shuffle(out,2)
        out = self.conv2(out)
        out = x + out
        # dp注意力机制代码
        out = self.dp_attention(out)
        # cp注意力机制
        #out = self.ca_attention(out)
        #out = self.pa_attention(out)
        out = out +x
        return out

# 定义block的组合
class Group(nn.Module):
    def __init__(self,in_planes,blocks):
        super(Group, self).__init__()
        modules = [ Block(in_planes)  for _ in range(blocks)]
        modules.append(conv3x3(in_planes,in_planes))
        modules.append(nn.ReLU())
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FSAM_Net(nn.Module):
    def __init__(self,in_planes):
        super(FSAM_Net,self).__init__()
        self.conv1 = conv3x3(in_planes,16) # 变成32通道
        self.norm1 = nn.InstanceNorm2d(16, affine=True)
        self.block1 = Block(16)
        self.conv2 = nn.Conv2d(16,32,3,2,3,3,bias=False)
        self.norm2 = nn.InstanceNorm2d(32,affine=True)
        self.block2 = Block(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 2, 2, bias=False)
        self.norm3 = nn.InstanceNorm2d(64,affine=True)

        self.group1 = Group(64,2)
        self.group2 = Group(64,2)
        self.group3 = Group(64,2)

        self.concat_con = conv1x1(64*3,64)

        self.deconv1 = nn.ConvTranspose2d(64,32,3,2,1,1)
        self.norm4 = nn.InstanceNorm2d(32,affine=True)
        self.block3 = Block(32)
        self.deconv2 = nn.ConvTranspose2d(32,16,3,2,1,1)
        self.norm5 = nn.InstanceNorm2d(16, affine=True)
        self.block4 = Block(16)
        self.conv4 = conv1x1(16,3) # 变回原通道3

        # 定义两个dplayer
        self.dp_layer1 = DP_layer(16)
        self.dp_layer2 = DP_layer(32)

    def forward(self,x):
        y1 = F.relu(self.norm1(self.conv1(x)))
        y2 = self.block1(y1) # 16 , 16 , 16
        y3 = F.relu((self.norm2(self.conv2(y2))))
        y4 = self.block2(y3) # 32 , 8, 8
        y5 = F.relu(self.norm3(self.conv3(y4)))
        y_group1 = self.group1(y5)  # 64 ,4 ,4
        y_group2 = self.group1(y_group1)  # 64 ,4 ,4
        y_group3 = self.group1(y_group2)  # 64 ,4 ,4
        conbine = torch.cat((y_group1,y_group2,y_group3),dim=1)
        conbine_result = self.concat_con(channel_shuffle(conbine,4))
        y7 = F.relu(self.norm4(self.deconv1(conbine_result)))
        #y7 = F.relu(self.norm4(self.deconv1(y_group3))
        y7 = y7 + self.dp_layer2(y4)
        y8 = self.block3(y7) # 32 , 8 ,8
        y9 = F.relu(self.norm5(self.deconv2(y8)))
        y9 = y9 + self.dp_layer1(y2)
        y10 = self.block4(y9) # 16,16,16
        y11 = self.conv4(y10)

        return y11


if __name__ == "__main__":
    N,C_in,H,W =  10,3,16,16
    x = torch.randn(N,C_in,H,W).float()
    y3 = FSAM_Net(3)
    result = y3(x)
    print(y3(x).shape)
    print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in y3.parameters()))
