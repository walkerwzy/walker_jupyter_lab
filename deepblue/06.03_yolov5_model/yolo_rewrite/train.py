import os
import cv2
import json
import math
import argparse

import data_provider
import dataset
import heads
import maptool
import models
import nn_utils
import sys_utils
import test

import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
from contextlib import contextmanager
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from sys_utils import _single_instance_logger as logger


@contextmanager
def torch_distributed_zero_first(local_rank, multi_gpu):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank != 0 and multi_gpu:
        torch.distributed.barrier()
    yield
    if local_rank == 0 and multi_gpu:
        torch.distributed.barrier()


class Config:
    def __init__(self):
        self.base_directory = "workspace"

    def get_path(self, path):
        return f"{self.base_directory}/{self.name}/{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)

    def get_train_dataset(self):

        if self.dataset == "VOC":
            provider = data_provider.VOCProvider("/data-rbd/wish/four_lesson/dataset/voc2007/VOCdevkitTrain/VOC2007")
        elif self.dataset == "COCO":
            provider = data_provider.COCOProvider("/data-rbd/wish/four_lesson/dataset/coco2017", "2017", "train")
        else:
            assert False, f"Unknow dataset {self.dataset}"

        return dataset.Dataset(True, config.image_size, provider)

    def get_test_dataset(self):

        if self.dataset == "VOC":
            provider = data_provider.VOCProvider("/data-rbd/wish/four_lesson/dataset/voc2007/VOCdevkitTest/VOC2007")
        elif self.dataset == "COCO":
            provider = data_provider.COCOProvider("/data-rbd/wish/four_lesson/dataset/coco2017", "2017", "val")
        else:
            assert False, f"Unknow dataset {self.dataset}"

        return dataset.Dataset(False, config.image_size, provider, config.total_batch_size)

    def get_model(self, train_dataset):
        num_classes = train_dataset.provider.num_classes
        model = models.Yolo(num_classes, self.network, self.local_rank)

        pixel_anchor = model.anchors * torch.FloatTensor(model.strides).view(3, 1, 1)
        new_anchor = nn_utils.fit_anchor(train_dataset.all_annotation_sizes, pixel_anchor.view(-1, 2).numpy())
        model.set_new_anchors(torch.FloatTensor(new_anchor).view(3, 3, 2))
        head = heads.YoloHead(num_classes, model.anchors, model.strides)
        return model, head

    def get_dataloader_with_dataset(self, dataset, istrain):

        if istrain:
            batch_size = config.batch_size
        else:
            batch_size = config.total_batch_size

        num_workers = min([os.cpu_count() // config.world_size, batch_size, 8])
        if self.multi_gpu and istrain:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    sampler=sampler,
                                                    pin_memory=True,
                                                    collate_fn=dataset.collate_fn)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate_fn)
        return dataloader

    def get_train_dataloader(self):
        with torch_distributed_zero_first(self.local_rank, self.multi_gpu):
            train_set = self.get_train_dataset()
            
        return self.get_dataloader_with_dataset(train_set, istrain=True)


    def get_test_dataloader(self):
        test_set = self.get_test_dataset()
        return self.get_dataloader_with_dataset(test_set, istrain=False)


def train():
    rank = config.local_rank
    nn_utils.setup_seed(2 + rank)
    sys_utils.copy_code_to(".", config.get_path("code"))

    torch.set_printoptions(linewidth=320, precision=5, profile='long')
    np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

    # Prevent OpenCV from multithreading (to use PyTorch DataLoader)
    cv2.setNumThreads(0)

    if rank == 0:
        if config.visdom:
            from visdom import Visdom
            visual_client = Visdom(server='http://127.0.0.1', port=8097, env=args.name)
            assert visual_client.check_connection(), "Connect to server failed"

        if config.tqdm:
            from tqdm import tqdm

    if config.amp:
        from apex import amp

    device = config.device
    train_dataloader = config.get_train_dataloader()
    if rank == 0:
        test_dataloader = config.get_test_dataloader()

    model, head = config.get_model(train_dataloader.dataset)
    if rank == 0:
        logger.info(f"Anchor is : \n{model.anchors}")

    best_model_score = 0.0
    momentum = 0.937
    weight_decay = 5e-4
    basic_number_of_batch_size = 64
    accumulate = max(round(basic_number_of_batch_size / config.total_batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= config.total_batch_size * accumulate / basic_number_of_batch_size 

    lr_start = 0.01
    otherwise_params, weight_and_not_bn_params, bias_params = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                bias_params.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                weight_and_not_bn_params.append(v)  # apply weight decay
            else:
                otherwise_params.append(v)  # all else
                
    optim = torch.optim.SGD(otherwise_params, lr=lr_start, momentum=momentum, nesterov=True)
    optim.add_param_group({'params': weight_and_not_bn_params, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optim.add_param_group({'params': bias_params})  # add pg2 (biases)
    del otherwise_params, weight_and_not_bn_params, bias_params
    model.to(device)
    head.to(device)

    if config.weight != "":
        if os.path.exists(config.weight):
            logger.info(f"Load weight: {config.weight}")
            check_point = torch.load(config.weight, map_location="cpu")

            model_dict = model.state_dict()
            check_point = {k : v for k, v in check_point.items() if k in model_dict and model_dict[k].shape == v.shape}
            mismatch_keys = [k for k, v in check_point.items() if k in model_dict.keys() and model_dict[k].shape != v.shape or k not in model_dict]
            mismatch_keys.extend([k for k, v in model_dict.items() if k in check_point.keys() and check_point[k].shape != v.shape or k not in check_point])

            if len(mismatch_keys) > 0:
                logger.info(f"Mismatch keys: {mismatch_keys}")

            # keep compute anchor
            check_point["anchors"] = model.anchors
            model.load_state_dict(check_point, strict=False)
            del check_point
        else:
            logger.error(f"Weight not exists: {config.weight}")

    if config.amp:
        model, optim = amp.initialize(model, optim, opt_level='O1', verbosity=0)

    #if config.multi_gpu:
    #    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    if rank == 0:
        ema = nn_utils.ModelEMA(model)

    if config.multi_gpu:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    epochs = config.epochs
    num_iter_per_epoch = len(train_dataloader)

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lf)
    nw = max(3 * num_iter_per_epoch, 1e3)
    
    for epoch in range(epochs):
        model.train()

        if config.multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
        
        pbar = enumerate(train_dataloader)
        if rank == 0:
            if config.tqdm:
                pbar = tqdm(pbar, total=num_iter_per_epoch)

        learning_rate = scheduler.get_last_lr()[0]
        optim.zero_grad()
        for batch_index, (images, targets, visual) in pbar:
            num_iter = batch_index + num_iter_per_epoch * epoch  # number integrated batches (since train start)

            # Warmup
            if num_iter <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(optim.param_groups):
                    accumulate = max(1, np.interp(num_iter, xi, [1, basic_number_of_batch_size / config.total_batch_size]).round())
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0

                    bias_param_group_index = 2
                    x['lr'] = np.interp(num_iter, xi, [0.1 if j == bias_param_group_index else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, [0.9, momentum])

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predicts = model(images)

            loss, loss_visual = head(predicts, targets)
            loss *= config.world_size

            if config.amp:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if num_iter % accumulate == 0:
                optim.step()
                optim.zero_grad()
                if rank == 0:
                    if ema is not None:
                        ema.update(model)

            if config.tqdm:
                if rank == 0:
                    num_targets = len(targets)
                    current_epoch = epoch + (batch_index + 1) / num_iter_per_epoch
                    log_line = f"Epoch: {current_epoch:.2f}/{epochs}, Iter: {num_iter}, Targets: {num_targets}, LR: {learning_rate:.5f}, {loss_visual}"
                    pbar.set_description(log_line)
            else:
                if rank == 0 and num_iter % 1000 == 0:
                    num_targets = len(targets)
                    current_epoch = epoch + (batch_index + 1) / num_iter_per_epoch
                    log_line = f"Iter: {num_iter}, Epoch: {current_epoch:.2f}/{epochs}, Targets: {num_targets}, LR: {learning_rate:.5f}, {loss_visual}"

                    logger.info(log_line)

                    if config.visdom:
                        objs = head.detect([item.detach() for item in predicts], confidence_threshold=0.5, nms_threshold=0.5)
                        visual_image_id, visual_image, visual_normalize_annotation, restore_info = visual
                        nn_utils.draw_norm_bboxes(visual_image, visual_normalize_annotation, color=(0, 0, 255))
                        nn_utils.draw_pixel_bboxes(visual_image, objs[visual_image_id].cpu().numpy(), color=(0, 255, 0))
                        visual_client.image(visual_image[..., ::-1].transpose(2, 0, 1), win="visual_image", opts={"title": "原始图"})
                        visual_client.text(log_line, win="logger", opts={"title": "日志"})

        scheduler.step()
        if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):

            current_epoch = int(current_epoch)
            saved_dict = None
            if ema is not None:
                saved_dict = ema.ema.state_dict()
                model_score = test.test(ema.ema, test_dataloader, head, current_epoch)
            else:
                if config.multi_gpu:
                    saved_dict = model.module.state_dict()
                else:
                    saved_dict = model.state_dict()
                model_score = test.test(model, test_dataloader, head, current_epoch)

            last_path = config.get_path("last.pth")
            best_path = config.get_path("best.pth")
            torch.save(saved_dict, last_path)

            if model_score > best_model_score:
                logger.info(f"Save [{current_epoch:03d} epoch] best model to {best_path}, model score is {model_score:.5f}, prev is {best_model_score:.5f}")
                best_model_score = model_score
                torch.save(saved_dict, best_path)

            del saved_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="实验名称", default="debug")
    parser.add_argument("--batch_size", type=int, help="批大小", default=16)
    parser.add_argument("--network", type=str, help="网络的配置文件", default="models/yolov5s.yaml")
    parser.add_argument("--weight", type=str, help="预训练模型", default="")   
    parser.add_argument("--dataset", type=str, help="数据集，可以选择VOC、COCO", default="COCO")
    parser.add_argument("--size", type=int, help="尺寸", default=640)
    parser.add_argument("--about", type=str, help="说明", default="")
    parser.add_argument("--epochs", type=int, help="轮数", default=30)
    parser.add_argument("--device", type=int, help="GPU号", default=1)
    parser.add_argument("--local_rank", type=int, help="local_rank", default=-1)
    parser.add_argument("--amp", action="store_true", help="使用自动混合精度")
    parser.add_argument("--tqdm", action="store_true", help="使用tqdm进度条展示")
    parser.add_argument("--visdom", action="store_true", help="使用visdom展示中间效果")
    args = parser.parse_args()

    config = Config()
    config.name = args.name
    
    config.multi_gpu = args.local_rank != -1
    if config.multi_gpu:
        config.local_rank = args.local_rank
        config.device = f"cuda:{config.local_rank}"
        torch.cuda.set_device(config.device)
        dist.init_process_group(backend='nccl', init_method='env://')
        config.world_size = dist.get_world_size()
    else:
        config.local_rank = 0
        config.device = f"cuda:{args.device}"
        torch.cuda.set_device(config.device)
        config.world_size = 1

    config.visdom = args.visdom
    config.tqdm = args.tqdm
    config.amp = args.amp
    config.image_size = args.size
    config.epochs = args.epochs
    config.weight = args.weight
    config.about = args.about
    config.network = args.network
    config.dataset = args.dataset
    config.total_batch_size = args.batch_size
    config.batch_size = config.total_batch_size // config.world_size
   
    sys_utils.setup_single_instance_logger(config.get_path("logs/log.log"))
    if config.local_rank == 0:
        logger.info(f"Startup, config: \n{config}")

    train()

    if config.multi_gpu and config.local_rank == 0:
        dist.destroy_process_group()
    
    torch.cuda.empty_cache()
