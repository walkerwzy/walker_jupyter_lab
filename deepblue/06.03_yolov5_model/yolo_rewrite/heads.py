import torch
import torchvision
import torch.nn as nn

class YoloHead(nn.Module):
    def __init__(self, num_classes, anchors, strides):
        super().__init__()
        '''
        anchors[3x3x2]   ->  level x scale x [width, height], 已经除以了stride后的anchor
        strides[list(3)] ->  [8, 16, 32]
        '''

        self.num_classes = num_classes
        self.strides = strides
        self.gap_threshold = 4 # anchor_t

        # 扩展样本时使用的边界偏移量
        self.offset_boundary = anchors.new_tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])

        # 3 scale x 3 anchor
        self.anchors = anchors
        self.num_anchor = 3
        self.reduction = "mean"
        self.loss_weight_giou_regression = 0.05
        self.loss_weight_objectness = 1.0
        self.loss_weight_classification = 0.5 * self.num_classes / 80
        self.balance = [4.0, 1.0, 0.4]  # 4, 8, 16 scales loss factor
        self.BCEClassification = nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.BCEObjectness = nn.BCEWithLogitsLoss(reduction=self.reduction)


    def to(self, device):
        super().to(device)
        self.anchors = self.anchors.to(device)
        self.offset_boundary = self.offset_boundary.to(device)
        return self

    
    def giou(self, a, b):
        '''
        计算a与b的GIoU
        参数：
        a[Nx4]：      要求是[cx, cy, width, height]
        b[Nx4]:       要求是[cx, cy, width, height]
        '''
        # a is n x 4
        # b is n x 4

        # cx, cy, width, height
        a_xmin, a_xmax = a[:, 0] - a[:, 2] / 2, a[:, 0] + a[:, 2] / 2
        a_ymin, a_ymax = a[:, 1] - a[:, 3] / 2, a[:, 1] + a[:, 3] / 2
        b_xmin, b_xmax = b[:, 0] - b[:, 2] / 2, b[:, 0] + b[:, 2] / 2
        b_ymin, b_ymax = b[:, 1] - b[:, 3] / 2, b[:, 1] + b[:, 3] / 2

        inter_xmin = torch.max(a_xmin, b_xmin)
        inter_xmax = torch.min(a_xmax, b_xmax)
        inter_ymin = torch.max(a_ymin, b_ymin)
        inter_ymax = torch.min(a_ymax, b_ymax)
        inter_width = (inter_xmax - inter_xmin).clamp(0)
        inter_height = (inter_ymax - inter_ymin).clamp(0)
        inter_area = inter_width * inter_height

        a_width, a_height = (a_xmax - a_xmin), (a_ymax - a_ymin)
        b_width, b_height = (b_xmax - b_xmin), (b_ymax - b_ymin)
        union = (a_width * a_height) + (b_width * b_height) - inter_area
        iou = inter_area / union

        # smallest enclosing box
        convex_width = torch.max(a_xmax, b_xmax) - torch.min(a_xmin, b_xmin) + 1e-16
        convex_height = torch.max(a_ymax, b_ymax) - torch.min(a_ymin, b_ymin)
        convex_area = convex_width * convex_height + 1e-16
        return iou - (convex_area - union) / convex_area

    # 上下计算结果是一样的，仅仅小数点后7位开始不同
    # def giou(self, a, b):
    #     '''
    #     计算a与b的GIoU
    #     参数：
    #     a[Nx4]：      要求是[cx, cy, width, height]，并且是pixel单位
    #     b[Nx4]:       要求是[cx, cy, width, height]，并且是pixel单位
    #     '''
    #     # a is n x 4
    #     # b is n x 4

    #     # cx, cy, width, height
    #     a_xmin, a_xmax = a[:, 0] - (a[:, 2] - 1) / 2, a[:, 0] + (a[:, 2] - 1) / 2
    #     a_ymin, a_ymax = a[:, 1] - (a[:, 3] - 1) / 2, a[:, 1] + (a[:, 3] - 1) / 2
    #     b_xmin, b_xmax = b[:, 0] - (b[:, 2] - 1) / 2, b[:, 0] + (b[:, 2] - 1) / 2
    #     b_ymin, b_ymax = b[:, 1] - (b[:, 3] - 1) / 2, b[:, 1] + (b[:, 3] - 1) / 2

    #     inter_xmin = torch.max(a_xmin, b_xmin)
    #     inter_xmax = torch.min(a_xmax, b_xmax)
    #     inter_ymin = torch.max(a_ymin, b_ymin)
    #     inter_ymax = torch.min(a_ymax, b_ymax)
    #     inter_width = (inter_xmax - inter_xmin + 1).clamp(0)
    #     inter_height = (inter_ymax - inter_ymin + 1).clamp(0)
    #     inter_area = inter_width * inter_height

    #     a_width, a_height = (a_xmax - a_xmin + 1), (a_ymax - a_ymin + 1)
    #     b_width, b_height = (b_xmax - b_xmin + 1), (b_ymax - b_ymin + 1)
    #     union = (a_width * a_height) + (b_width * b_height) - inter_area
    #     iou = inter_area / union

    #     # smallest enclosing box
    #     convex_width = torch.max(a_xmax, b_xmax) - torch.min(a_xmin, b_xmin) + 1
    #     convex_height = torch.max(a_ymax, b_ymax) - torch.min(a_ymin, b_ymin) + 1
    #     convex_area = convex_width * convex_height
    #     return iou - (convex_area - union) / convex_area

    def forward(self, predict, targets):
        # bbox is [image_id, classes_id, cx, cy, width, height]
        # targets[num_targets, bbox]
        device = targets.device
        loss_classification = torch.FloatTensor([0]).to(device)
        loss_box_regression = torch.FloatTensor([0]).to(device)
        loss_objectness = torch.FloatTensor([0]).to(device)
        num_target = targets.shape[0]

        for ilayer, layer in enumerate(predict):
            layer_height, layer_width = layer.shape[-2:]

            # batch, num_anchors, height, width, 6[x, y, r, b, objectness, classess]
            layer = layer.view(-1, 3, 5 + self.num_classes, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()

            # image_id, classes_id, cx, cy, width, height
            # targets is NumTarget x 6
            layer_anchors = self.anchors[ilayer]
            num_anchor = layer_anchors.shape[0]
            feature_size_gain = targets.new_tensor([1, 1, layer_width, layer_height, layer_width, layer_height])
            targets_feature_scale = targets * feature_size_gain
            anchors_wh = layer_anchors.view(num_anchor, 1, 2)
            targets_wh = targets_feature_scale[:, 4:6].view(1, num_target, 2)

            # # wh_ratio is [num_anchor, num_target, 2]
            wh_ratio = targets_wh / anchors_wh

            # # select_mask is [num_anchor, num_target]
            max_wh_ratio_values, _ = torch.max(wh_ratio, 1 / wh_ratio).max(2)
            select_mask = max_wh_ratio_values < self.gap_threshold

            # NumTarget x num_anchor, 1
            # target -> anchor
            # targets.repeat(num_anchor, 1, 1) -> num_anchor x NumTarget x 6
            # select_targets is [matched_num_target, 6]
            select_targets = targets_feature_scale.repeat(num_anchor, 1, 1)[select_mask]
            matched_num_target = len(select_targets)

            featuremap_objectness = layer[..., 4]
            objectness_ground_truth = torch.zeros_like(featuremap_objectness, device=device)

            if matched_num_target > 0:
                #  0  0  0  0  0  0  0  ...   num_target
                #  1  1  1  1  1  1  1 
                #  2  2  2  2  2  2  2
                # anchor_index_repeat is [num_anchor, num_target]
                anchor_index_repeat = torch.arange(num_anchor, device=device).view(num_anchor, 1).repeat(1, num_target)

                # select_anchor_index is [matched_num_target]
                select_anchor_index = anchor_index_repeat[select_mask]

                # 扩展采样，在原本中心的位置上增加采样点，根据中心坐标，x、y距离谁近，选择一个谁
                # 这里不考虑cx, cy正好为0.5的情况，则等于将样本增加2倍
                # select_targets_xy is [matched_num_target, 2]
                select_targets_xy = select_targets[:, 2:4]
                xy_divided_one_remainder = select_targets_xy % 1.0
                coord_cell_middle = 0.5
                feature_map_low_boundary = 1.0
                feature_map_high_boundary = feature_size_gain[[2, 3]] - 1.0
                less_x_matched, less_y_matched = ((xy_divided_one_remainder < coord_cell_middle) & (select_targets_xy > feature_map_low_boundary)).t()
                greater_x_matched, greater_y_matched = ((xy_divided_one_remainder > (1 - coord_cell_middle)) & (select_targets_xy < feature_map_high_boundary)).t()

                select_anchor_index = torch.cat([
                    select_anchor_index, 
                    select_anchor_index[less_x_matched],
                    select_anchor_index[less_y_matched],
                    select_anchor_index[greater_x_matched],
                    select_anchor_index[greater_y_matched]
                ], dim=0)

                select_targets = torch.cat([
                    select_targets, 
                    select_targets[less_x_matched],
                    select_targets[less_y_matched],
                    select_targets[greater_x_matched],
                    select_targets[greater_y_matched]
                ], dim=0)

                xy_offsets = torch.zeros_like(select_targets_xy, device=device)
                xy_offsets = torch.cat([
                    xy_offsets, 
                    xy_offsets[less_x_matched] + self.offset_boundary[0],
                    xy_offsets[less_y_matched] + self.offset_boundary[1],
                    xy_offsets[greater_x_matched] + self.offset_boundary[2],
                    xy_offsets[greater_y_matched] + self.offset_boundary[3]
                ]) * coord_cell_middle

                # image_id, classes_id, cx, cy, width, height
                # .t()的目的是把nx2转置为2xn，可以直接解包为2个变量，一个变量为一行
                matched_extend_num_target = select_targets.shape[0]
                gt_image_id, gt_classes_id = select_targets[:, :2].long().t()
                gt_xy = select_targets[:, 2:4]
                gt_wh = select_targets[:, 4:6]
                grid_xy = (gt_xy - xy_offsets).long()
                grid_x, grid_y = grid_xy.t()
                gt_xy = gt_xy - grid_xy

                # select_anchors is [matched_extend_num_target, 2]
                select_anchors = layer_anchors[select_anchor_index]

                #####################################################
                # object_position_predict is [matched_extend_num_target, 6]
                object_position_predict = layer[gt_image_id, select_anchor_index, grid_y, grid_x]
                object_predict_xy = (object_position_predict[:, :2].sigmoid() * 2.0 - 0.5).float()
                object_predict_wh = torch.pow(object_position_predict[:, 2:4].sigmoid() * 2.0,  2.0) * select_anchors

                # matched_extend_num_target, 4
                object_predict_box = torch.cat((object_predict_xy, object_predict_wh), dim=1)
                object_ground_truth_box = torch.cat((gt_xy, gt_wh), dim=1)
                gious = self.giou(object_predict_box, object_ground_truth_box)
                giou_loss = 1.0 - gious
                loss_box_regression += giou_loss.mean() if self.reduction == "mean" else giou_loss.sum()
                objectness_ground_truth[gt_image_id, select_anchor_index, grid_y, grid_x] = gious.detach().clamp(0).type(objectness_ground_truth.dtype)
                
                # gious = gious.detach().clamp(0).type(objectness_ground_truth.dtype)
                # unique_value, inverse_index, value_counts = torch.stack([gt_image_id, select_anchor_index, grid_y, grid_x], dim=0).unique(dim=1, return_counts=True, return_inverse=True)
                # if unique_value.size(1) != matched_extend_num_target:
                #     item_counts = value_counts[inverse_index]
                #     item_indexs = torch.arange(len(value_counts), device=device)[inverse_index]
                #     unique_ious = torch.zeros(value_counts.shape[0], device=device, dtype=gious.dtype)
                #     for item_index, unique_index in enumerate(item_indexs):
                #         unique_ious[unique_index] = torch.max(gious[item_index], unique_ious[unique_index])
                #     objectness_ground_truth[torch.unbind(unique_value)] = unique_ious
                # else:
                #     objectness_ground_truth[gt_image_id, select_anchor_index, grid_y, grid_x] = gious

                if self.num_classes > 1:
                    # classification loss (only if multiple classes)
                    # object_classification is [matched_extend_num_target, num_classes]
                    object_classification = object_position_predict[:, 5:]
                    classification_targets = torch.full_like(object_classification, 0, device=device)
                    classification_targets[torch.arange(matched_extend_num_target), gt_classes_id] = 1.0
                    loss_classification += self.BCEClassification(object_classification, classification_targets)

            # batch, num_anchors, height, width, 6[x, y, r, b, objectness, classess]
            # batch, num_anchors, height, width
            loss_objectness += self.BCEObjectness(featuremap_objectness, objectness_ground_truth) * self.balance[ilayer]

        num_predict = len(predict)
        scale = 3 / num_predict
        batch_size = predict[0].shape[0]
        loss_box_regression *= self.loss_weight_giou_regression * scale
        loss_objectness *= self.loss_weight_objectness * scale * (1.4 if num_predict == 4 else 1.0)
        loss_classification *= self.loss_weight_classification * scale
        
        loss = loss_box_regression + loss_objectness + loss_classification
        loss_visual = f"Loss: {loss.item():.06f}, Box: {loss_box_regression.item():.06f}, Obj: {loss_objectness.item():.06f}, Cls: {loss_classification.item():.06f}"
        return loss * batch_size, loss_visual


    def detect(self, predict, confidence_threshold=0.3, nms_threshold=0.5, multi_table=True):
        '''
        检测目标
        参数：
        predict[layer8, layer16, layer32],      每个layer是BxCxHxW
        confidence_threshold，                  保留的置信度阈值
        nms_threshold，                         nms的阈值
        '''
        batch = predict[0].shape[0]
        device = predict[0].device
        objs = []
        for ilayer, (layer, stride) in enumerate(zip(predict, self.strides)):
            layer_height, layer_width = layer.size(-2), layer.size(-1)
            layer = layer.view(batch, 3, 5 + self.num_classes, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()
            layer = layer.sigmoid().view(batch, 3, -1, layer.size(-1))
            
            if self.num_classes == 1:
                object_score = layer[..., 4]
                object_classes = torch.zeros_like(object_score)
                keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            else:
                layer_confidence = layer[..., [4]] * layer[..., 5:]
                if multi_table:
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes = torch.where(layer_confidence > confidence_threshold)
                    object_score = layer_confidence[keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes]
                else:
                    object_score, object_classes = layer_confidence.max(-1)
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            
            num_keep_box = len(keep_batch_indices)
            if num_keep_box == 0:
                continue

            keepbox = layer[keep_batch_indices, keep_anchor_indices, keep_cell_indices].float()
            layer_anchors = self.anchors[ilayer]
            keep_anchors = layer_anchors[keep_anchor_indices]
            cell_x = keep_cell_indices % layer_width
            cell_y = keep_cell_indices // layer_width
            keep_cell_xy = torch.cat([cell_x.view(-1, 1), cell_y.view(-1, 1)], dim=1)
            wh_restore = (torch.pow(keepbox[:, 2:4] * 2, 2) * keep_anchors) * stride
            xy_restore = (keepbox[:, :2] * 2.0 - 0.5 + keep_cell_xy) * stride
            object_score = object_score.float().view(-1, 1)
            object_classes = object_classes.float().view(-1, 1)
            keep_batch_indices = keep_batch_indices.float().view(-1, 1)
            box = torch.cat((keep_batch_indices, xy_restore - (wh_restore - 1) * 0.5, xy_restore + (wh_restore - 1) * 0.5, object_score, object_classes), dim=1)
            objs.append(box)

        if len(objs) > 0:
            objs_cat = torch.cat(objs, dim=0)
            objs_image_base = []
            for ibatch in range(batch):
                # left, top, right, bottom, score, classes
                select_box = objs_cat[objs_cat[:, 0] == ibatch, 1:]
                objs_image_base.append(select_box)
        else:
            objs_image_base = [torch.zeros((0, 6), device=device) for _ in range(batch)]
        
        if nms_threshold is not None:
            # 使用类内的nms，类间不做操作
            for ibatch in range(batch):
                image_objs = objs_image_base[ibatch]
                if len(image_objs) > 0:
                    max_wh_size = 4096
                    classes = image_objs[:, [5]]
                    bboxes = image_objs[:, :4] + (classes * max_wh_size)
                    confidence = image_objs[:, 4]
                    keep_index = torchvision.ops.boxes.nms(bboxes, confidence, nms_threshold)
                    objs_image_base[ibatch] = image_objs[keep_index]
        return objs_image_base