import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml

import nn_utils
from sys_utils import _single_instance_logger as logger

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def autopad(kernel, padding=None):  # kernel, padding
    # Pad to 'same'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]  # auto-pad
    return padding


class Conv(nn.Module):
    '''
    标准卷积层
    '''
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, autopad(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        # 合并后的前向推理，bn和卷积合并
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    '''
    标准瓶颈层
    '''
    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        hidden_channel = int(out_channel * expansion)  # hidden channels
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        hidden_channel = int(out_channel * expansion)  # hidden channels
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1, bias=False)
        self.cv4 = Conv(2 * hidden_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_channel)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion=1.0) for _ in range(repeats)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernel_size_list=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_channel = in_channel // 2  # hidden channels
        self.cv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.cv2 = Conv(hidden_channel * (len(kernel_size_list) + 1), out_channel, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) for kernel_size in kernel_size_list])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super(Focus, self).__init__()
        self.conv = Conv(in_channel * 4, out_channel, kernel_size, stride, padding, groups, activation)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.d)


class Detect(nn.Module):
    def __init__(self, num_classes, num_anchor, reference_channels):
        super(Detect, self).__init__()
        self.num_anchor = num_anchor
        self.num_classes = num_classes
        self.num_output = self.num_classes + 5
        self.m = nn.ModuleList(nn.Conv2d(input_channel, self.num_output * self.num_anchor, 1) for input_channel in reference_channels)
        self.init_weight()

    def forward(self, x):
        for ilevel, module in enumerate(self.m):
            x[ilevel] = module(x[ilevel])
        return x

    def init_weight(self):
        strides = [8, 16, 32]
        for head, stride in zip(self.m, strides):
            bias = head.bias.view(self.num_anchor, -1)
            bias[:, 4] += math.log(8 / (640 / stride) ** 2)
            bias[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            head.bias = nn.Parameter(bias.view(-1), requires_grad=True)


class Yolo(nn.Module):
    def __init__(self, num_classes, config_file, rank=0):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.rank = rank
        self.strides = [8, 16, 32]
        self.model, self.saved_index, anchors = self.build_model(config_file)
        self.register_buffer("anchors", torch.FloatTensor(anchors).view(3, 3, 2) / torch.FloatTensor(self.strides).view(3, 1, 1))
        self.apply(self.init_weight)

    
    def set_new_anchors(self, anchors):
        # 对设置的anchors缩放到特征图大小
        self.anchors[...] = anchors / torch.FloatTensor(self.strides).view(3, 1, 1)


    def init_weight(self, m):
        type_t = type(m)
        if type_t is nn.Conv2d:
            # pass init
            pass
        elif type_t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif type_t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
    
    def forward(self, x):
        y = []
        for module in self.model:
            if module.from_index != -1:
                if isinstance(module.from_index, int):
                    x = y[module.from_index]
                else:
                    xout = []
                    for i in module.from_index:
                        if i == -1:
                            xval = x
                        else:
                            xval = y[i]
                        xout.append(xval)
                    x = xout
            
            x = module(x)
            y.append(x if module.layer_index in self.saved_index else None)
        return x

    def parse_string(self, value):
        if value == "None":
            return None
        elif value == "True":
            return True
        elif value == "False":
            return False
        else:
            return value

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = nn_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuse_forward  # update forward
        return self
    
    def build_model(self, config_file, input_channel=3):

        with open(config_file) as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)

        all_layers_cfg_list = self.yaml["backbone"] + self.yaml["head"]
        anchors, depth_multiple, width_multiple = [self.yaml[item] for item in ["anchors", "depth_multiple", "width_multiple"]]
        num_classes = self.num_classes
        num_anchor = len(anchors[0]) // 2
        num_output = num_anchor * (num_classes + 5)
        all_layers_channels = [input_channel]
        all_layers = []
        saved_layer_index = []

        for layer_index, (from_index, repeat_count, module_name, args) in enumerate(all_layers_cfg_list):
            args = [self.parse_string(a) for a in args]
            module_function = eval(module_name)

            if repeat_count > 1:
                repeat_count = max(round(repeat_count * depth_multiple), 1)
            
            if module_function in [Conv, Bottleneck, SPP, Focus, BottleneckCSP]:
                channel_input, channel_output = all_layers_channels[from_index], args[0]

                if channel_output != num_output:
                    channel_output = make_divisible(channel_output * width_multiple, 8)

                args = [channel_input, channel_output, *args[1:]]
                if module_function in [BottleneckCSP]:
                    args.insert(2, repeat_count)
                    repeat_count = 1
            
            elif module_function is Concat:
                channel_output = sum([all_layers_channels[-1 if x == -1 else x + 1] for x in from_index])
            elif module_function is Detect:
                reference_channel = [all_layers_channels[x + 1] for x in from_index]
                args = [num_classes, num_anchor, reference_channel]
            else:
                channel_output = all_layers_channels[from_index]

            if repeat_count > 1:
                module_instance = nn.ModuleList([
                    module_function(*args) for _ in range(repeat_count)
                ])
            else:
                module_instance = module_function(*args)

            module_instance.from_index = from_index
            module_instance.layer_index = layer_index
            all_layers.append(module_instance)
            all_layers_channels.append(channel_output)

            if not isinstance(from_index, list):
                from_index = [from_index]
            saved_layer_index.extend(filter(lambda x: x!=-1, from_index))

            num_params = sum([x.numel() for x in module_instance.parameters()])

            if self.rank == 0:
                align_format = "%6s %-15s %-7s %-10s %-18s %-30s"

                if layer_index == 0:
                    logger.info(align_format % ("Index", "From", "Repeats", "Param", "Module", "Arguments"))

                format_vals = (
                    "%d." % layer_index,
                    str(from_index),
                    str(repeat_count),
                    "%d"  % num_params,
                    module_name,
                    str(args)
                )
                logger.info(align_format % format_vals)

        return nn.Sequential(*all_layers), sorted(saved_layer_index), anchors


if __name__ == "__main__":
    import nn_utils

    nn_utils.setup_seed(3)

    device = "cuda:0"
    model = Yolo(20, "/datav/wish/yolov5/models/yolov5s.yaml").to(device)
    model.fuse()

    checkpoint = torch.load("/datav/wish/yolov5-2.0/test.pt", map_location="cpu")
    checkpoint['anchors'] = checkpoint['model.24.anchors']
    del checkpoint['model.24.anchors']
    del checkpoint['model.24.anchor_grid']
    model.load_state_dict(checkpoint)
    print("Done")
    # weight = "/datav/wish/yolov5/weights/yolov5m.pt"

    # import pickle

    # with open(weight, "rb") as f:
    #     p = pickle.Unpickler(f, fix_imports=False)
    #     value = p.load()
    #     print(value)
    #check_point = torch.load(weight, map_location="cpu", fix_imports=False)
    #print(check_point)

    #input = torch.zeros((1, 3, 640, 640))
    #output = model(input)
    #print(output[0].shape)