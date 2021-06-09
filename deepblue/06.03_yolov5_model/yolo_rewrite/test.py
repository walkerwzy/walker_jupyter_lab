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

import torch
import torch.nn as nn

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


    def get_test_dataset(self):

        if self.dataset == "VOC":
            provider = data_provider.VOCProvider("/data-rbd/wish/four_lesson/dataset/voc2007/VOCdevkitTest/VOC2007")
        elif self.dataset == "COCO":
            provider = data_provider.COCOProvider("/data-rbd/wish/four_lesson/dataset/coco2017", "2017", "val")
        else:
            assert False, f"Unknow dataset {self.dataset}"

        return dataset.Dataset(False, config.image_size, provider, config.batch_size)


    def get_dataloader_with_dataset(self, dataset):

        batch_size = config.batch_size
        num_workers = min([os.cpu_count(), batch_size, 8])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=dataset.collate_fn)
        return dataloader



def test(model, test_loader, head, epoch=0):
    model.eval()
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    batch_size = test_loader.batch_size

    with torch.no_grad():

        groundtruth_annotations = {}
        detection_annotations = {}

        # test loader是使用centerAffine进行的
        # normalize_annotations格式是[image_id, class_index, cx, cy, width, height]
        for batch_index, (images, normalize_annotations, visual) in enumerate(tqdm(test_loader, desc=f"Eval map {epoch:03d} epoch")):
            images = images.to(device, non_blocking=True).type(dtype)
            predicts = model(images)

            # 检测目标，得到的结果是[left, top, right, bottom, confidence, classes]
            objects = head.detect(predicts, confidence_threshold=0.001, nms_threshold=0.6)
            
            batch, channels, image_height, image_width = images.shape
            visual_image_id, visual_image, visual_annotations, restore_info = visual

            num_batch = images.shape[0]
            normalize_annotations = normalize_annotations.to(device)
            restore_info = normalize_annotations.new_tensor(restore_info) # pad_left, pad_top, origin_width, origin_height, scale

            pixel_annotations = nn_utils.convert_to_pixel_annotation(normalize_annotations[:, [2, 3, 4, 5, 0, 1]], image_width, image_height)
            for i in range(num_batch):
                index = torch.where(pixel_annotations[:, 4] == i)[0]
                if len(index) == 0:
                    continue
                
                padx, pady, origin_width, origin_height, scale = restore_info[i]
                pixel_annotations[index, :4] = (pixel_annotations[index, :4] - restore_info[i, [0, 1, 0, 1]]) / scale
            
            for left, top, right, bottom, image_id, class_id in pixel_annotations.cpu().numpy():
                image_id = int(image_id) + batch_index * batch_size
                class_id = int(class_id)
                if image_id not in groundtruth_annotations:
                    groundtruth_annotations[image_id] = []

                groundtruth_annotations[image_id].append([left, top, right, bottom, 0, class_id])

            for image_index, image_objs in enumerate(objects):
                image_objs[:, 0].clamp_(0, image_width)
                image_objs[:, 1].clamp_(0, image_height)
                image_objs[:, 2].clamp_(0, image_width)
                image_objs[:, 3].clamp_(0, image_height)

                padx, pady, origin_width, origin_height, scale = restore_info[image_index]
                image_objs[:, :4] = (image_objs[:, :4] - restore_info[image_index, [0, 1, 0, 1]]) / scale
                image_id = image_index + batch_index * batch_size
                detection_annotations[image_id] = image_objs.cpu().numpy()

        # merge groundtruth_annotations
        for image_id in groundtruth_annotations:
            groundtruth_annotations[image_id] = np.array(groundtruth_annotations[image_id], dtype=np.float32)

        map_result = maptool.MAPTool(groundtruth_annotations, detection_annotations, test_loader.dataset.provider.label_map)
        map05, map075, map05095 = map_result.map
        model_score = map05 * 0.1 + map05095 * 0.9
        logger.info(f"Eval {epoch:03d} epoch, mAP@.5 [{map05:.6f}], mAP@.75 [{map075:.6f}], mAP@.5:.95 [{map05095:.6f}], Time: {map_result.compute_time:.2f} second")
    return model_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="实验名称", default="debug")
    parser.add_argument("--batch_size", type=int, help="批大小", default=32)
    parser.add_argument("--network", type=str, help="网络的配置文件", default="models/yolov5m.yaml")
    parser.add_argument("--weight", type=str, help="权重", default="/datav/wish/yolov5-2.0/cocom.pt")
    parser.add_argument("--dataset", type=str, help="数据集", default="COCO")
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
    config.dataset = args.dataset
    config.batch_size = args.batch_size
   
    sys_utils.setup_single_instance_logger(config.get_path("logs/log.log"))
    logger.info(f"Startup, config: \n{config}")

    checkpoint = torch.load(config.weight, map_location="cpu")
    from_yolo_raw_checkpoint = "model.24.anchors" in checkpoint

    dataset = config.get_test_dataset()
    num_classes = dataset.provider.num_classes
    model = models.Yolo(num_classes, config.network)
    model.eval()
    model.to(config.device)

    if from_yolo_raw_checkpoint:
        logger.info("Use yolo raw checkpoint")
        checkpoint['anchors'] = checkpoint['model.24.anchors']
        del checkpoint['model.24.anchors']
        del checkpoint['model.24.anchor_grid']
        head = heads.YoloHead(num_classes, model.anchors, model.strides)

        model.fuse()
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
        head = heads.YoloHead(num_classes, model.anchors, model.strides)
        model.fuse()
    
    model.half()
    dataloader = config.get_dataloader_with_dataset(dataset)
    test(model, dataloader, head)
