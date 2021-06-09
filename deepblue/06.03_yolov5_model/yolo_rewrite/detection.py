import os
import cv2
import json
import argparse

import data_provider
import dataset
import heads
import maptool
import models
import nn_utils
import sys_utils
import math

import torch
import torch.nn as nn
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sys_utils import _single_instance_logger as logger

class Config:
    def __init__(self):
        self.base_directory = "workspace"
        self.batch_size = 32
        self.name = "default"      # 实验名称


    def get_path(self, path):
        return f"{self.base_directory}/{self.name}/{path}"


    def __repr__(self):
        return json.dumps(self.__dict__, indent=4, ensure_ascii=False)


def preprocess_load_image(file, image_size):

    image = cv2.imread(file)
    image_height, image_width = image.shape[:2]
    to_image_size_ratio = image_size / max(image_width, image_height)
    if to_image_size_ratio != 1:
        image = cv2.resize(image, dsize=(0, 0), fx=to_image_size_ratio, fy=to_image_size_ratio)

    stride = 32
    ih, iw = image.shape[:2]
    oh = int(math.ceil(ih / stride)) * stride
    ow = int(math.ceil(iw / stride)) * stride

    pad_left = (ow - iw) // 2
    pad_top = (oh - ih) // 2
    pad_right = ow - iw - pad_left
    pad_bottom = oh - ih - pad_top
    image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR -> RGB
    rgb_image = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)) 

    # normalize
    rgb_image = rgb_image / np.array([255.0], dtype=np.float32)
    return torch.from_numpy(rgb_image).unsqueeze(dim=0), image


def detect(model, head, files, image_size, saveto):

    model.eval()
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    with torch.no_grad():

        # test loader是使用centerAffine进行的
        # normalize_annotations格式是[image_id, class_index, cx, cy, width, height]
        for file_index, path in enumerate(tqdm(files, desc=f"Inference: ")):
            torch_image, image = preprocess_load_image(path, image_size)

            torch_image = torch_image.to(device, non_blocking=True).type(dtype)
            predicts = model(torch_image)

            # 检测目标，得到的结果是[left, top, right, bottom, confidence, classes]
            objects = head.detect(predicts, confidence_threshold=0.25, nms_threshold=0.5)
            
            batch, channels, image_height, image_width = torch_image.shape
            num_batch = torch_image.shape[0]
            for image_index, image_objs in enumerate(objects):
                image_objs[:, 0].clamp_(0, image_width)
                image_objs[:, 1].clamp_(0, image_height)
                image_objs[:, 2].clamp_(0, image_width)
                image_objs[:, 3].clamp_(0, image_height)
                
                nn_utils.draw_pixel_bboxes(image, image_objs)

            save_path = f"{saveto}/{Path(path).name}"
            cv2.imwrite(save_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="实验名称", default="debug")
    parser.add_argument("--network", type=str, help="网络的配置文件", default="/datav/wish/yolov5/models/yolov5m.yaml")
    parser.add_argument("--weight", type=str, help="权重", default="workspace/coco_yolov5m/best.pth")
    parser.add_argument("--source", type=str, help="数据源", default="/data-rbd/wish/four_lesson/dataset/coco2017/test2017")
    parser.add_argument("--saveto", type=str, help="数据源", default="inference/detect")
    parser.add_argument("--size", type=int, help="尺寸", default=640)
    parser.add_argument("--device", type=int, help="GPU号", default=1)
    args = parser.parse_args()

    config = Config()
    config.name = args.name
    
    config.device = f"cuda:{args.device}"
    torch.cuda.set_device(config.device)

    config.image_size = args.size
    config.weight = args.weight
    config.network = args.network
    config.source = args.source
    config.saveto = args.saveto
   
    sys_utils.setup_single_instance_logger(config.get_path("logs/log.log"))
    logger.info(f"Startup, config: \n{config}")

    checkpoint = torch.load(config.weight, map_location="cpu")
    from_yolo_raw_checkpoint = "model.24.anchors" in checkpoint

    model = models.Yolo(80, config.network)
    model.eval()
    model.to(config.device)

    if from_yolo_raw_checkpoint:
        logger.info("Use yolo raw checkpoint")
        checkpoint['anchors'] = checkpoint['model.24.anchors']
        del checkpoint['model.24.anchors']
        del checkpoint['model.24.anchor_grid']
        head = heads.YoloHead(80, model.anchors, model.strides)

        model.fuse()
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        head = heads.YoloHead(80, model.anchors, model.strides)
        model.fuse()
    
    model.half()

    files = os.listdir(config.source)
    files = [f"{config.source}/{file}" for file in files]
    np.random.shuffle(files)
    files = files[:10]
    sys_utils.mkdirs(config.saveto)
    detect(model, head, files, config.image_size, config.saveto)
