import os
import sys
import cv2
import torch
import random
import sys_utils
import nn_utils
import numpy as np
import traceback

from tqdm import tqdm

from pycocotools.coco import COCO
from sys_utils import _single_instance_logger as logger
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
import json

# 格式定义：
# pixel_annotations：       像素为单位的标注，格式是[left, top, right, bottom]，绝对位置标注
# normalize_annotations:    归一化后的标注0-1，除以图像宽高，格式是[cx, cy, width, height]

# class Provider:
#     def __len__(self):
#         raise NotImplementedError()

#     def __getitem__(self, index):
#         raise NotImplementedError()


class Provider:
    def __init__(self, cache_file):
        self.all_labeled_information = []
        self.build_and_cache(cache_file)


    # 要求返回的是：  生成器, label_map，total[用来显示进度的]
    # 生成器返回值是：jpeg_file, pixel_annotations[left, top, right, bottom, class_index]
    def build_image_and_annotations_generate(self):
        raise NotImplementedError()


    def build_and_cache(self, cache_file):
        if os.path.exists(cache_file):
            logger.info(f"Load labels from cache: {cache_file}")
            self.load_labeled_information_from_cache(cache_file)
        else:
            logger.info(f"Build labels and save to cache: {cache_file}")
            self.build_labeled_information_and_save(cache_file)

    
    def load_labeled_information_from_cache(self, cache_file):
        '''
        从缓存文件中加载标注信息
        '''
        self.all_labeled_information, self.label_map = torch.load(cache_file)


    def build_labeled_information_and_save(self, cache_file):
        '''
        构建数据集，存储图像路径和标注信息，采用多线程进行
        '''
        miss_files = 0
        image_file_and_pixel_annotations_generate, label_map, total_files = self.build_image_and_annotations_generate()
        pbar = tqdm(image_file_and_pixel_annotations_generate, total=total_files, desc="检索图像信息中")
        thread_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="prefix_")
        miss_file_log = None
        miss_file_json = []
        miss_file_lock = Lock()

        def process_file(jpeg_file, pixel_annotations):
            nonlocal miss_files, miss_file_log, miss_file_lock

            try:
                # 数据检查
                # 1. 图像是否损坏，如果损坏，直接抛异常
                # 2. 检查图像大小是否过小，如果太小，直接异常
                # 加载标注信息，并保存起来
                #    标注信息是normalize过的

                # 做一个定义
                # 1. 如果基于像素单位的框，绝对位置框，定义为pixel类
                # 2. 如果是归一化后（除以图像宽高的归一化）的框，定义为normalize类
                pil_image = Image.open(jpeg_file)
                # 暂时没有exif

                # 如果图像不正常，损坏，他会直接给你抛异常
                pil_image.verify()

                image_width, image_height = sys_utils.exif_size(pil_image)
                assert image_width > 9 and image_height > 9, f"Image size is too small{image_width} x {image_height}"
            except Exception as e:
                
                miss_file_lock.acquire()
                if miss_file_log is None:
                    sys_utils.mkparents(cache_file)
                    miss_file_log = open(f"{cache_file}.miss.log", "w")

                miss_file_json.append([jpeg_file, repr(e)])
                miss_file_log.write(traceback.format_exc())
                miss_file_log.flush()
                miss_files += 1
                miss_file_lock.release()
                return None
            
            # 转换到normalize，同时变为[cx, cy, width, height]
            normalize_annotations = nn_utils.convert_to_normalize_annotation(pixel_annotations, image_width, image_height)
            return [jpeg_file, normalize_annotations, [image_width, image_height]]

        result_futures = []
        for jpeg_file, pixel_annotations in pbar:
            result_futures.append(thread_pool.submit(process_file, jpeg_file, pixel_annotations))
            pbar.set_description(f"Search and cache, total = {total_files}, miss = {miss_files}")

        self.all_labeled_information = []
        for item in result_futures:
            result = item.result()
            if result is not None:
                self.all_labeled_information.append(result)

        if miss_file_log is not None:
            miss_file_log.write(json.dumps(miss_file_json, indent=4, ensure_ascii=False) + "\n")
            miss_file_log.close()
        
        self.label_map = label_map
        sys_utils.mkparents(cache_file)
        torch.save([self.all_labeled_information, self.label_map], cache_file)

    
    @property
    def num_classes(self):
        return len(self.label_map)


    def __len__(self):
        return len(self.all_labeled_information)


    def __getitem__(self, image_indice):
        jpeg_file, normalize_annotations, (image_width, image_height) = self.all_labeled_information[image_indice]
        image = cv2.imread(jpeg_file)
        return image, normalize_annotations, (image_width, image_height)


class VOCProvider(Provider):
    '''
    VOC的数据提供者
    '''
    def __init__(self, root, cache_root="dataset_cache"):
        # /datav/shared/db/voc2007/VOCdevkitTrain/VOC2007/
        # /datav/shared/db/voc2007/VOCdevkitTest/VOC2007/
        self.root = root

        cache_name = sys_utils.get_md5(root)
        super().__init__(f"{cache_root}/voc_{cache_name}.cache")

    
    def build_image_and_annotations_generate(self):
        # 生成器, label_map，total[用来显示进度的]
        annotations_files = os.listdir(os.path.join(self.root, "Annotations"))

        # 保留所有的xml后缀文件
        annotations_files = list(filter(lambda x: x.endswith(".xml"), annotations_files))
        total_files = len(annotations_files)

        # xml改jpg
        jpeg_files = [item[:-3] + "jpg" for item in annotations_files]
        
        # 把文件名修改为全路径
        annotations_files = map(lambda x: os.path.join(self.root, "Annotations", x), annotations_files)
        jpeg_files = map(lambda x: os.path.join(self.root, "JPEGImages", x), jpeg_files)
        
        label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        def generate_function():
            for jpeg_file, annotation_file in zip(jpeg_files, annotations_files):
                pixel_annotations = nn_utils.load_voc_annotation(annotation_file, label_map)
                yield jpeg_file, pixel_annotations

        return generate_function(), label_map, total_files


class COCOProvider(Provider):
    '''
    COCO数据的提供者
    '''
    def __init__(self, root, year="2017", prefix="train", cache_root="dataset_cache"):
        self.root = root
        self.year = year
        self.prefix = prefix
        self.annotation_file = os.path.join(self.root, 'annotations', f'instances_{self.prefix}{self.year}.json')
        cache_name = sys_utils.get_md5(self.annotation_file)
        super().__init__(f"{cache_root}/coco_{cache_name}.cache")


    def build_image_and_annotations_generate(self):
        # 生成器, label_map，total[用来显示进度的]
        self.annotation_file = os.path.join(self.root, 'annotations', f'instances_{self.prefix}{self.year}.json')

        coco = COCO(self.annotation_file)
        image_ids = coco.getImgIds()
        categories = coco.loadCats(coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        class_name_to_index        = {}
        class_index_to_name        = {}
        class_index_to_category_id = {}
        category_id_to_class_index = {}
        label_map                  = []
        for class_index, category in enumerate(categories):
            category_id = category['id']
            category_name = category['name']
            class_index_to_category_id[class_index] = category_id
            category_id_to_class_index[category_id] = class_index
            class_name_to_index[category_name] = class_index
            class_index_to_name[class_index] = category_name
            label_map.append(category_name)

        miss_files = 0
        total_files = len(image_ids)

        def generate_function():
            for image_index in range(len(image_ids)):
                image_id   = image_ids[image_index]
                image_info = coco.loadImgs(image_id)[0]
                jpeg_file  = os.path.join(self.root, f"{self.prefix}{self.year}", image_info['file_name'])
                pixel_annotations = self.load_annotations(coco, category_id_to_class_index, image_id)
                yield jpeg_file, pixel_annotations

        return generate_function(), label_map, total_files


    # annotations is [bbox]
    # bbox is left, top, right, bottom, classes
    def load_annotations(self, coco, category_id_to_class_index, image_id):
        # get ground truth annotations
        annotations_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
        pixel_annotations = np.zeros((0, 5), dtype=np.float32)

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return pixel_annotations

        # parse annotations
        coco_annotations = coco.loadAnns(annotations_ids)
        annotations_list = []
        for idx, coco_annotation in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            left, top, width, height = coco_annotation['bbox']
            category_id = coco_annotation['category_id']
            if width < 2 or height < 2:
                continue
            
            class_index  = category_id_to_class_index[category_id]
            annotations_list.append([left, top, left+width-1, top+height-1, class_index])

        if len(annotations_list) > 0:
            pixel_annotations = np.array(annotations_list, dtype=np.float32)
        return pixel_annotations


if __name__ == "__main__":
    #provider = COCOProvider("/data-rbd/wish/four_lesson/dataset/coco2017/")
    provider = VOCProvider("/data-rbd/wish/four_lesson/dataset/voc2007/VOCdevkitTrain/VOC2007/")
    print(len(provider))
    print(provider.num_classes)
    print(provider.label_map)

    provider = COCOProvider("/data-rbd/wish/four_lesson/dataset/coco2017/")
    print(len(provider))
    print(provider.num_classes)
    print(provider.label_map)


