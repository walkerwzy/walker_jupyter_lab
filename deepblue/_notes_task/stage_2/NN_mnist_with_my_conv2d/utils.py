import numpy as np
import struct
import random
# import matplotlib.pyplot as plt
# import pandas as pd
import math
import time

def cal_time(func):
    '''timer'''
    def improved_func(*args):
        start_time = time.time()
        res = func(*args)
        end_time = time.time()
        print('{} takes {}'.format(func.__name__,end_time-start_time))
        return res
    return improved_func
        
def set_seed(seed):
    import random
    import os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_samples = struct.unpack(">ii", data[:8])
    if magic_number != 2049:  # 0x00000801
        print(f"magic number mismatch {magic_number} != 2049")
        return None

    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def load_images(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_samples, image_width, image_height = struct.unpack(">iiii", data[:16])
    if magic_number != 2051:  # 0x00000803
        print(f"magic number mismatch {magic_number} != 2051")
        return None

    image_data = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_samples, -1)
    return image_data


def one_hot(labels, classes, label_smoothing=0):
    n = len(labels)
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    for row, label in enumerate(labels):
        output[row, label] = 1 - label_smoothing + eoff
    return output

class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    # 获取他的一个item，  dataset = Dataset(),   dataset[index]
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    # 获取数据集的长度，个数
    def __len__(self):
        return len(self.images)


class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cursor = 0
        self.indexs = list(range(self.dataloader.count_data))  # 0, ... 60000
        if self.dataloader.shuffle:
            # 打乱一下
            random.shuffle(self.indexs)

    def __next__(self):
        if self.cursor >= self.dataloader.count_data:
            raise StopIteration()

        batch_data = []
        remain = min(self.dataloader.batch_size, self.dataloader.count_data - self.cursor)  # 256, 128
        for n in range(remain):
            index = self.indexs[self.cursor]
            data = self.dataloader.dataset[index]  
            '''
            data is a tuple consisting of image and labels
                def __getitem__(self, index):
                    return self.images[index], self.labels[index]
            '''

            # 如果batch没有初始化，则初始化n个list成员
            if len(batch_data) == 0:
                batch_data = [[] for i in range(len(data))]  # batch_data = [[for imgs ][for one_hot labels ]]
            '''
            fmt: [ [arr1, arr2], [oh_label1, oh_label2] ]
            '''
            # 直接append进去
            for index, item in enumerate(data):
                batch_data[index].append(item)

            self.cursor += 1

        # 通过np.vstack一次性实现合并，而非每次一直在合并
        for index in range(len(batch_data)):  # index0 imgs  index1 labels.
            # batch_data[index] = np.vstack(batch_data[index])
            batch_data[index] = np.array(batch_data[index])
        return batch_data
        '''
        batch_data = [imgs_arr(64,1,28,28),  labels_arr(64,10)]
        '''


class DataLoader:

    # shuffle 打乱
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.shuffle = shuffle
        self.count_data = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        return DataLoaderIterator(self)


def estimate(plabel, gt_labels, classes):
    plabel = plabel.copy()
    gt_labels = gt_labels.copy()
    match_mask = plabel == classes
    mismatch_mask = plabel != classes
    plabel[match_mask] = 1
    plabel[mismatch_mask] = 0

    gt_mask = gt_labels == classes
    gt_mismatch_mask = gt_labels != classes
    gt_labels[gt_mask] = 1
    gt_labels[gt_mismatch_mask] = 0

    TP = sum(plabel & gt_labels)
    FP = sum(plabel & (1 - gt_labels))
    FN = sum((1 - plabel) & gt_labels)
    TN = sum((1 - plabel) & (1 - gt_labels))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, F1


def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == val_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))


def lr_cosine_schedule(lr_min, lr_max, Ti):
    '''
    :param Ti: Ti epochs are performed before a new restart.
    :param Tcur: How many epochs have been performed since the last restart.
    :return: a function to compute a value within a period.
    '''

    def compute(Tcur):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(Tcur / Ti * np.pi))

    return compute


def sigmoid(x):
    p0 = x < 0
    p1 = ~p0  # 补集
    x = x.copy()
    x[p0] = np.exp(x[p0]) / (np.exp(x[p0]) + 1)
    x[p1] = 1 / (1 + np.exp(-x[p1]))
    return x


def softmax(z):
    '''
    softmax.png
    e.g. z: (64, 10)
    '''
    z = z.copy()
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    res = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return res


def cross_entropy(predict, gt):
    eps = 1e-4
    predict = np.clip(predict, a_max=1 - eps, a_min=eps)  # 裁切
    batch_size = predict.shape[0]
    return -np.sum(gt * np.log(predict) + (1 - gt) * np.log(1 - predict)) / batch_size  # loss for one batch


class Module:
    def __init__(self, name):
        self.name = name
        self.train_mode = False

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.train_mode = True
        for m in self.modules():
            m.train()

    def eval(self):
        self.train_mode = False
        for m in self.modules():
            m.eval()

    def modules(self):
        ms = []
        for attr in self.__dict__:
            m = self.__dict__[attr]
            if isinstance(m, Module):
                ms.append(m)
        return ms

    def params(self):
        ps = []
        for attr in self.__dict__:
            p = self.__dict__[attr]
            if isinstance(p, Parameter):
                ps.append(p)

        ms = self.modules()
        for m in ms:
            ps.extend(m.params())
        return ps

    def info(self, n):
        ms = self.modules()
        output = f"{self.name}\n"
        for m in ms:
            output += (' ' * (n + 1)) + f"{m.info(n + 1)}\n"
        return output[:-1]

    def __repr__(self):
        return self.info(0)

class ModuleList(Module):
    def __init__(self, *args):
        super().__init__("ModuleList")
        self.ms = list(args)

    def __repr__(self):
        return f'self.ms'

    def modules(self):
        return self.ms

    def forward(self, x):
        for m in self.ms:
            x = m(x)
        return x

    def backward(self, G):
        for i in range(len(self.ms) - 1, -1, -1):
            G = self.ms[i].backward(G)
        return G

class Model(Module):
    def __init__(self):
        super().__init__("Model")
        self.backbone = ModuleList(
            Conv2d(in_feature = 1, out_feature = 5, kernel_size = 3),
            ReLU(),
            Flatten(),
            Linear(input_feature = 3380, output_feature = 10)
        )

        # self.backbone = ModuleList(
        #     Du_Conv2d(in_feature = 1, out_feature = 5, kernel_size = 3),
        #     ReLU(),
        #     Flatten(),
        #     Linear(input_feature = 3380, output_feature = 10)
        # )

    def forward(self, x):   
        return self.backbone(x)

    def backward(self, G):
        return self.backbone.backward(G)

class Initializer:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        return self.apply(*args)

class GaussInitializer(Initializer):
    # where :math:`\mu` is the mean and :math:`\sigma` the standard
    # deviation. The square of the standard deviation, :math:`\sigma^2`,
    # is called the variance.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def apply(self, value):
        value[...] = np.random.normal(self.mu, self.sigma, value.shape)

class Parameter:
    def __init__(self, value):
        self.value = value  # self.value 和 value 在内存里的reference 是一样的，意味着修改value就相当于修改self.value
        self.delta = np.zeros(value.shape)

    def zero_grad(self):
        self.delta[...] = 0

class Linear(Module):
    def __init__(self, input_feature, output_feature):
        super().__init__("Linear")
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weights = Parameter(np.zeros((input_feature, output_feature)))
        self.bias = Parameter(np.zeros((1, output_feature)))

        # 权重初始化
        initer = GaussInitializer(0, np.sqrt(2 / input_feature))  # kaiming初始化
        initer.apply(self.weights.value)

    def forward(self, x): # (2, 4, 26, 26)
        self.x_save = x.copy()
        return x @ self.weights.value + self.bias.value

    # AB = C  G
    # dB = A.T @ G
    # dA = G @ B.T
    def backward(self, G):
        self.weights.delta += self.x_save.T @ G
        # +=是因为考虑了多个batch后再更新；这里不用/batch_size 是因为回传的第一个G
        # 也就是loss 的G 已经除以了batchsize 了。
        self.bias.delta += np.sum(G, 0)  # 值复制
        return G @ self.weights.value.T

class Kernel():
    def __init__(self,arr,stride = 1,padding = 0):
        '''
        arr :[out_fea,in_fea,ksize,ksize] a.k.a [groups,channels,ksize,ksize]。
        下面讨论的都是kernel 的绝对坐标，我们不关心kernel内元素的相对坐标。
        arr只有成为class Kenrnel的instance才能拥有以下kernel的所有功能
        '''    
        self.arr = arr
        self.groups,self.channels,self.ksize = self.arr.shape[:3]    # kernel的组数，kernel的通道数，kernel的大小
        self.stride = stride
        self.padding = padding
        
        self.st_offset = self.ksize//2            # (inclusive) img的左上角坐标直接加上start_offset就可以得到kernel在图片上的起点
        self.ed_offset = -(self.ksize//2)         # (exclusive) img的右下角直接加上这个end_offset就可以得到kernel在图片上的终点
        self.ele_num_each_channel = self.ksize**2 # kernel每个channel的元素个数
 
   
    def get_tl(self,center):                      # 通过kernel的中心坐标获取kernel的左上角坐标（wrt img）
        cy,cx = center
        y,x = cy-self.ksize//2,cx-self.ksize//2   
        return y,x
    
    def get_covered_pixels(self,center,itensor):  # 获取kernel覆盖到的pixels given center and itensor.
        y,x = self.get_tl(center) 
        pixels = itensor[:,:,y:y+self.ksize,x:x+self.ksize]     # 将所有图片，所有通道，kernel覆盖到的像素拿出来
        return pixels
    
    def to_kcol(self):                                          # 将kernel变成kcol
        return self.arr.reshape(self.groups,-1)
    
    def get_output_img_size(self,itensor):
        _,_,ih,iw = itensor.shape
        oh = (ih-self.ksize + 2*self.padding)//self.stride + 1
        ow = (iw-self.ksize + 2*self.padding)//self.stride + 1
        return oh,ow

class Conv2d(Module):
    def __init__(self, in_feature, out_feature, kernel_size, padding=0, stride=1):
        super().__init__("Conv2d")
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.ksz = kernel_size
        self.stride = stride
        self.padding = padding
        # self.kernel = np.array([  # only used for unit test
        #                     [0,0,0],
        #                     [1,1,0],
        #                     [0,0,0]
        #             ])[None][None]
        self.knl_arr = np.zeros((self.out_feature, self.in_feature, self.ksz, self.ksz))
        self.knl_obj = Kernel(self.knl_arr, stride=self.stride, padding=self.padding) # make kernel_arr to be a real kernel(knl_obj). As soon as we got the knl_obj, we mainly communicate with knl_obj.
        self.kcol = self.knl_obj.to_kcol() # (4,9)
        
        self.weight = Parameter(self.knl_obj.arr)  # Parameter instance
        self.bias = Parameter(np.zeros((out_feature)))

        initer = GaussInitializer(0, np.sqrt(2 / in_feature))  # np.sqrt(2 / input_feature)
        initer.apply(self.weight.value)
        pass
    
    # @cal_time
    def forward(self, x):
        kcol = self.kcol
        knl_obj = self.knl_obj

        # 开始构建column（挪kernel)
        self.in_shape = x.shape
        ib, ic, ih, iw = self.in_shape
        self.oh, self.ow = knl_obj.get_output_img_size(x)
        self.column = np.zeros((ib, knl_obj.ksize ** 2 * ic, self.oh * self.ow)) # (2,9,676)
        self.output = np.zeros((ib, self.out_feature, self.oh ,self.ow)) # (2,4,26,26)

        # start to carry pixels from itensor to column
        # cy,cx 是kernel的中心
        j = 0  # 用来表明是column的第几列
        for cy in range(knl_obj.st_offset, ih + knl_obj.ed_offset):
            for cx in range(knl_obj.st_offset, iw + knl_obj.ed_offset):
                pixels = knl_obj.get_covered_pixels((cy, cx), x)
                cols = pixels.reshape(ib, knl_obj.ksize ** 2 * knl_obj.channels, -1)  # 将多张图片的coverer_pixels放到column对应的列里去（涵盖三个通道）
                self.column[:, :, None, j] = cols
                j += 1

        output = kcol @ self.column # (2,4,676)
        self.output = output.reshape(ib, knl_obj.groups, self.oh, self.ow) + self.bias.value.reshape(self.out_feature,1,1)

        return self.output

    def backward(self, G): # 这个G已经除以了batchsize
        # In im2col, column @ kcol = output
        # dcolumn = G @ kcol^T
        # dkcol = column^T @ G   G: d_L/d_output
        ib, ic, ih, iw = self.in_shape
        knl_obj = self.knl_obj
        # 1. update part

        self.weight.delta = np.sum((G.reshape(-1, self.out_feature, self.oh * self.ow) @ self.column.transpose(0, 2, 1)), axis = 0).reshape(self.knl_arr.shape)
        self.bias.delta += np.sum(G, axis=(0, 2, 3))  # 因为G的第一个通道是out_feature,对应的就是有多少组kernel

        # 2. pass-back part
        self.Gout = np.zeros(self.in_shape)  # because Gout is d_L/d_input

        dcolumn = self.kcol.T @ G.reshape(-1, self.out_feature, self.oh * self.ow)  # "one" input image --> v output image. If the output channel is v.
        # this is the thing to be passed back.
        j = 0  # indexing the j-th col in dcolumn starts from 0
        for cy in range(knl_obj.st_offset, ih + knl_obj.ed_offset):  # the part you need to fill gradients back into dcolumn
            for cx in range(knl_obj.st_offset, iw + knl_obj.ed_offset):
                y,x = self.knl_obj.get_tl((cy,cx))
                self.Gout[:, :, y:y+self.ksz, x:x+self.ksz] += dcolumn[:,:,j].reshape(ib, self.in_feature, self.ksz, self.ksz)
                j+=1

        return self.Gout

class Du_Conv2d(Module):
    def __init__(self,in_feature, out_feature, kernel_size, padding = 0, stride = 1):
        super().__init__("Conv2d")
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.kernel = Parameter(np.zeros((out_feature, in_feature, kernel_size, kernel_size)))# 就是这样定义的
        # self.kernel = Parameter(
        #             np.array([
        #                     [0,0,0],
        #                     [1,1,0],
        #                     [0,0,0] 
        #             ])[None][None])
        self.bias = Parameter(np.zeros((out_feature)))# 每一组kernel 配一个bias
        initer = GaussInitializer(0,2/np.sqrt(in_feature))
#         initer.apply(self.kernel.value)

    @cal_time 
    def forward(self,x):
        # the input :img and kernel
        self.in_shape = x.shape
        ib,ic,ih,iw = self.in_shape
        self.khalf = self.kernel_size//2
        # output
        self.oh = (ih-self.kernel_size + 2*self.padding)//self.stride + 1
        self.ow = (iw-self.kernel_size + 2*self.padding)//self.stride + 1
        self.output = np.zeros((ib,self.out_feature,self.oh,self.ow))
    
        # column
        self.column = np.zeros((ib,self.kernel_size*self.kernel_size*ic ,self.oh*self.ow))
        # k_col
        self.k_col = self.kernel.value.reshape((self.out_feature,-1))

        
        for b in range(ib):
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value goes in column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value comes from img
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.column[b,column_y, column_x] = x[b,channel,iy,ix]
            
            self.output[b] = (self.k_col @ self.column[b]).reshape(-1,self.oh,self.ow) + self.bias.value.reshape((self.out_feature,1,1))       
        return self.output  
        
    def backward(self,G):# G : G_in : dL/d output(this layer)
        ib,ic,ih,iw = self.in_shape # the shape of x  [input of the current layer]
        
        # 1.update part
        # k_col @ column = output
        for b in range(ib):
            # 首先三维的G[b] 肯定是要reshape成2维。因为G[b]：d output(this layer)，所以shape与output[b]是一样的
            # output[b]是[out_feature,oh,ow]
            self.kernel.delta += (G[b].reshape(-1,self.oh*self.ow)@self.column[b].T).reshape(self.kernel.value.shape) # column[b].T shape: (oh*ow,kh*kw*channel)
        
        self.bias.delta += np.sum(G,axis = (0,2,3)) # 因为G的第一个通道是out_feature,对应的就是有多少组kernel
        
        # 2.pass back part
        self.Gout = np.zeros((self.in_shape))

        for b in range(ib):
            # dcolumn我们这里仅仅作为当前图片的dcolumn
            dcolumn = self.k_col.T @ G[b].reshape(self.out_feature,-1) # k_col.T shape: (kw*kh*ic,out_feature)
            # dcolumn 和column shape是一样的
  
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value comes from column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value goes to img 可参考 notability 笔记 “输入输出坐标的推导”
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.Gout[b,channel,iy,ix] += dcolumn[column_y, column_x]
                                    #上面之所以使用+= 是因为在im2col的时候，一个img像素会搬到column的多个地方
                                    #（由于是滑动窗口会重叠），也就是说一个像素会在column不同地方出现，所以回传的时候
                                    #有多个地方贡献梯度
        
        return self.Gout


class ReLU(Module):
    def __init__(self, inplace=True):
        super().__init__("ReLU")
        self.inplace = inplace

    def forward(self, x):
        self.x_negative = x < 0
        if not self.inplace:
            x = x.copy()

        x[self.x_negative] = 0
        return x

    def backward(self, G):
        if not self.inplace:
            G = G.copy()

        G[self.x_negative] = 0
        return G

class Maxpool(Module):
    def __init__(self, kernel_size = 2, stride = 1):
        super().__init__("Maxpool")
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, x): # not considering padding
        self.in_shape = x.shape
        ib, ic, ih, iw = x.shape
        self.oh = int(np.ceil((ih - self.ksize)/self.stride + 1))  # Using ceil here amounts to inexplictly padding so that size can be exact divided by stride.
        self.ow = int(np.ceil((iw - self.ksize)/self.stride + 1))
        output = np.zeros((ib,ic,self.oh,self.ow))

        for oy in range(self.oh):
            for ox in range(self.ow):
                output[:,:, oy, ox] = np.max(x[:, :, oy*self.stride: oy*self.stride + self.ksize,
                                                     ox*self.stride: ox*self.stride + self.ksize ])

            '''
            No need to worry about the even or odd value of the size.
                let's say we have an x with size (64, 16, 11, 11), given ksz = 2, stride = 2
                11 = 2x5 + 1

                the first 5 indexing will return 
                array([[0., 0.],
                       [0., 0.]])

                But the last one will return 
                array([[0.],
                       [0.]])
            '''
        return output
        pass

    def backward(self, G): 


        # return Gout
        pass



class Flatten(Module):
    # flatten仅仅只是将输入一维化，不影响batch大小
    def __init__(self):
        super().__init__("Flatten")

    def forward(self, x):
        self.in_shape = x.shape
        self.out = x.reshape(self.in_shape[0], -1)  # 保留batch大小不变
        return self.out

    def backward(self, G):  # G : dL/dx 所以跟x是一个形状
        return G.reshape(self.in_shape)

class Dropout(Module):
    def __init__(self, pro_keep=0.5, inplace=True):
        super().__init__("Dropout")
        self.pro_keep = pro_keep
        self.inplace = inplace

    def forward(self, x):
        if not self.train_mode:
            return x

        self.mask = np.random.binomial(size=x.shape, p=1 - self.pro_keep, n=1)
        if not inplace:
            x = x.copy()
        x[self.mask] = 0  # 压制住每层false的输入神经元
        x *= 1 / self.pro_keep  # 需要rescale
        return x

    def backward(self, G):
        if not self.train_mode:
            return G

        if not inplace:
            G = G.copy()

        G[self.mask] = 0
        G *= 1 / self.pro_keep
        return G

class SigmoidCrossEntropy(Module):
    def __init__(self):
        super().__init__("CrossEntropyLoss")

    def forward(self, x, one_hot_labels):
        self.labels = one_hot_labels
        self.predict = sigmoid(x)
        self.batch_size = self.predict.shape[0]
        loss = cross_entropy(self.predict, self.labels) / self.batch_size  # loss for one batch
        return loss

    def backward(self):
        return (self.predict - self.labels) / self.batch_size

class SoftmaxCrossEntropy(Module):
    def __init__(self):
        """
        aewgawegaewgaewg
            awegawegaweg
            softmax.png
        """
        super().__init__("SoftmaxCrossEntropy")

    def forward(self, x, one_hot_labels):
        self.predict = softmax(x)
        self.labels = one_hot_labels
        self.batch_size = self.predict.shape[0]
        loss = cross_entropy(self.predict, self.labels) / self.batch_size  # loss for one batch
        return loss
    
    def backward(self):
        return (self.predict - self.labels) / self.batch_size

class Optimizer:
    def __init__(self, name, model, lr):
        self.name = name
        self.model = model
        self.lr = lr
        self.params = model.params()

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def set_lr(self, lr):
        self.lr = lr

class SGD(Optimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__("SGD", model, lr)

    def step(self):
        for param in self.params:
            param.value -= self.lr * param.delta

class Adam(Optimizer):  # l2 和adam不要一起用 https://zhuanlan.zhihu.com/p/63982470
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999):
        super().__init__("Adam", model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

        for param in self.params:
            param.m = 0  # w和b参数多了m v两个属性
            param.v = 0

    def step(self):
        eps = 1e-8
        self.t += 1
        for param in self.params:
            g = param.delta
            param.m = self.beta1 * param.m + (1 - self.beta1) * g
            param.v = self.beta2 * param.v + (1 - self.beta2) * g ** 2

            param.m_ = param.m / (1 - self.beta1 ** self.t)
            param.v_ = param.v / (1 - self.beta2 ** self.t)

            param.value -= self.lr * param.m_ / (np.sqrt(param.v_) + eps)


def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == gt_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))

def lr_schedule_cosine(lr_min, lr_max, per_epochs):
    def compute(epoch):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))

    return compute

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # UNIT TEST CONV2D
    # itensor = np.arange(784).reshape(28,28)[None]
    # itensor = np.stack((itensor, itensor))

    # set_seed(3)
    # conv2d = Conv2d(in_feature=1,out_feature=4,kernel_size=3,padding = 0)
    # # forward
    # print("my forward")
    # output = conv2d(itensor)

    # # backward
    # print("my backward")
    # G = output
    # Gout = conv2d.backward(G)
    # print("Done")


    # ----------------------------------------------------------------------
    # UNIT TEST CONV2D IN A TOY NN
    itensor = np.arange(784).reshape(28,28)[None]
    itensor = np.stack((itensor, itensor))
    labels = np.zeros((2, 10))
    toy_nn = Model() # for mnist
    loss_func = SoftmaxCrossEntropy()
    optim = SGD(toy_nn, lr = 0.1)

    toy_nn.backbone = ModuleList(
            Conv2d(in_feature = 1, out_feature = 4, kernel_size = 3),
            Flatten(),
            Linear(input_feature = 2704, output_feature = 10)
        )

    x = toy_nn(itensor)
    # 计算loss值
    loss = loss_func(x, labels)
    G = loss_func.backward()
    toy_nn.backward(G)
    optim.step()

    print("Well Done!")
    
    
