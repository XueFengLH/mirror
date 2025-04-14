# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import logging, math
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from typing import List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss
from .loss import build_set_criterion
from .head import build_dynamic_head
from .matcher import build_matcher, build_matcher_1
from config import config
from torchvision.transforms.functional import crop
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou_kk
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized)
from torchvision.ops.boxes import box_area
from .dcnv3 import build_deformable_conv3

__all__ = ["SparseRCNN"]


@META_ARCH_REGISTRY.register()
class SparseRCNN(nn.Module):
    """
    Implement SparseRCNN
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.mix1 = nn.Sequential(MixStructureBlock(256))
        self.mix2 = nn.Sequential(MixStructureBlock(256))
        self.mix3 = nn.Sequential(MixStructureBlock(256))
        self.mix4 = nn.Sequential(MixStructureBlock(256))
        self.back_l = [self.mix1,self.mix2,self.mix3,self.mix4]
        self.size_divisibility = self.backbone.size_divisibility

        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # Build Dynamic Head.
        self.head = build_dynamic_head(cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
        giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
        l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
        no_object_weight = cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = build_matcher(cfg, class_weight, l1_weight, giou_weight, self.use_focal)
        self.matcher = build_matcher(cfg, class_weight, l1_weight, giou_weight, self.use_focal)
        self.matcher_1 = build_matcher_1(cfg, class_weight, l1_weight, giou_weight, self.use_focal)

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = build_set_criterion(cfg=cfg,
                                             num_classes=self.num_classes,
                                             matcher=matcher,
                                             weight_dict=weight_dict,
                                             eos_coef=no_object_weight,
                                             losses=losses,
                                             use_focal=self.use_focal)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        no_fan = False
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        if no_fan:
            images, images_whwh = self.preprocess_image(batched_inputs)
            if isinstance(images, (list, torch.Tensor)):
                images = nested_tensor_from_tensor_list(images)

            # Feature Extraction.
            src = self.backbone(images.tensor)
            features = [src[f] for f in self.in_features]
            # draw_features1(features,batched_inputs[0]['image'],batched_inputs[0]['file_name'])
            # dcnv3
            ff = []
            for f, p in zip(features, self.back_l):
                ff.append(p(f))
            features = ff
            # draw_features(features,batched_inputs[0]['image'],batched_inputs[0]['file_name'])
            # Prepare Proposals.
            proposal_boxes = self.init_proposal_boxes.weight.clone()
            proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
            proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

            # Prediction.
            outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_ignore = [x["ignore"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, gt_ignore)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            sx = False
            th = 0.9
            if no_fan:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                results = self.inference(box_cls, box_pred, images.image_sizes)
            results_fan_zhuan = self.forward_3(batched_inputs[0]['image'], sx=sx,file_name= batched_inputs[0]['file_name'])
            img_width = batched_inputs[0]['image'].shape[2]
            img_hight = batched_inputs[0]['image'].shape[1]
            hui_fu = results_fan_zhuan[0].pred_boxes.tensor
            # hui_fu_1 = results[0].pred_boxes.tensor
            # hui_fu_1[:, [0, 2]] = torch.abs(hui_fu_1[:, [2, 0]] - img_width)
            if sx:
                hui_fu[:, [1, 3]] = torch.abs(hui_fu[:, [3, 1]] - img_hight)
            else:
                hui_fu[:, [0, 2]] = torch.abs(hui_fu[:, [2, 0]] - img_width)
            # boxes_cls_pred = torch.cat([results[0].pred_boxes.tensor, torch.unsqueeze(results[0].scores, 1)], 1)
            boxes_cls_pred_fan = torch.cat([hui_fu, torch.unsqueeze(results_fan_zhuan[0].scores, 1)], 1)
            # boxes_cls_pred_1 = torch.cat([hui_fu_1, torch.unsqueeze(results[0].scores, 1)], 1)
            # boxes_cls_pred = self.pai_xu(boxes_cls_pred, boxes_cls_pred[:, 4])
            # boxes_cls_pred_fan = self.pai_xu(boxes_cls_pred_fan,boxes_cls_pred_fan[:,4])
            # boxes_cls_pred_mask = boxes_cls_pred[:, 4] > 0.4
            boxes_cls_pred_fan_mask = boxes_cls_pred_fan[:, 4] > 0
            # boxes_cls_pred_1 = boxes_cls_pred[boxes_cls_pred_mask]
            boxes_cls_pred_fan_1 = boxes_cls_pred_fan[boxes_cls_pred_fan_mask]
            # index_1 = self.matcher_1(boxes_cls_pred_fan_1, boxes_cls_pred_1)
            # index_1 = self.matcher_1(boxes_cls_pred_1,boxes_cls_pred_fan_1)

            # 修改数据
            # if index_1 != 0:
            #     for i,j in enumerate(index_1[0][1]):
            #         xxx= boxes_cls_pred_fan[i][None, :]
            #         kiou = box_iou_kk(boxes_cls_pred_fan[i][None,:], boxes_cls_pred[j][None,:])
            #         if boxes_cls_pred_fan[i][4] < boxes_cls_pred[j][4] and kiou > 0.8:
            #
            #             boxes_cls_pred_fan[i] = boxes_cls_pred[j]

            # r_max,index = self.f1(boxes_cls_pred)
            # if r_max > 1:
            #     print(r_max)
            #     boxes_cls_pred[index+1:,:] = 0
            #
            # r_max,index = self.f1(boxes_cls_pred_fan)
            # if r_max > 0.40:
            #     print(r_max)
            #     boxes_cls_pred_fan[index+1:,:] = 0

            # #计算kiou
            # box_kiou = box_iou_kk(boxes_cls_pred, boxes_cls_pred_fan)

            # masks = box_kiou > th
            # #获取boxes——list
            # boxes_list = []
            # for i,m in enumerate(masks):
            #     box_d = {}
            #     box_d['box_cls_pred'] = boxes_cls_pred[i]
            #     if len(boxes_cls_pred_fan[m]) > 1:
            #         # box_d['list'] = self.pai_xu(boxes_cls_pred_fan[m],boxes_cls_pred_fan[:,4][m])
            #         pai = self.pai_xu(torch.arange(500).to(boxes_cls_pred_fan.device)[m], boxes_cls_pred_fan[:, 4][m])
            #         box_d['list_id'] = pai.tolist()
            #     boxes_list.append(box_d)
            # j = 0
            # pi_pei = []
            # for bl in boxes_list:
            #     if bl['box_cls_pred'][4] < 0.4:
            #         break
            #     if len(bl) == 1:
            #         boxes_cls_pred_fan[len(boxes_cls_pred_fan)-j-1] = bl['box_cls_pred']
            #         print(boxes_cls_pred_fan[len(boxes_cls_pred_fan) - j - 1])
            #         pi_pei.append(len(boxes_cls_pred_fan)-j-1)
            #         j = j + 1
            #         continue
            #     if len(pi_pei) != 0:
            #         ids = [x for x in bl['list_id'] if x not in pi_pei]
            #     else:
            #         ids = bl['list_id']
            #     if len(ids) != 0:
            #         if boxes_cls_pred_fan[ids[0]][4] < bl['box_cls_pred'][4]:
            #             boxes_cls_pred_fan[i] = bl['box_cls_pred']
            #             print(boxes_cls_pred_fan[i])
            #             pi_pei.append(ids[0])

            #
            # print(1)

            # for b_k,b_c_p in box_kiou,boxes_cls_pred:
            #     kiou_pai_xu = self.pai_xu(boxes_cls_pred_fan,b_k)
            # boxes_cls_pred = boxes_cls_pred[:100]
            # images_whwh_500 = images_whwh.squeeze(0).repeat(len(boxes_cls_pred))
            # images_whwh_500 = images_whwh_500.reshape(len(boxes_cls_pred),4)
            #
            #
            # output = {'pred_logits': boxes_cls_pred[None,:,:4], 'pred_boxes': boxes_cls_pred[None,:,4,None]}
            # output_fan = {'pred_logits': boxes_cls_pred_fan[None,:,4,None], 'pred_boxes': boxes_cls_pred_fan[None,:,:4]}
            # output_target = [{'labels':torch.zeros(len(boxes_cls_pred),dtype=torch.int),'boxes_xyxy':boxes_cls_pred[:,:4],'boxes':boxes_cls_pred[:,:4],'image_size_xyxy_tgt':images_whwh_500 ,'image_size_xyxy':images_whwh.squeeze(0) }]
            # xx = self.matcher(output_fan,output_target)
            pass
            # 输出图片

            # draw_box(batched_inputs[0]['file_name'],boxes_cls_pred,batched_inputs[0]['image'].shape)
            # 图片输出
            draw_box(batched_inputs[0]['file_name'], boxes_cls_pred_fan, batched_inputs[0]['image'].shape,fanzhuan=False)
            # # zong_he = torch.cat([boxes_cls_pred_fan,boxes_cls_pred],)
            # # threshold = 0.8
            # # keep = torchvision.ops.nms(zong_he[:,:4], zong_he[:,4], threshold)
            # # zong_he = zong_he[keep]

            pass
            results = self.inference_1(boxes_cls_pred_fan[:, 4][None, :, None], boxes_cls_pred_fan[None, :, :4],
                                       images.image_sizes)
            # results = self.inference_1(boxes_cls_pred[:,4][None,:,None], boxes_cls_pred[None,:,:4], images.image_sizes)
            # results = self.inference_1(boxes_cls_pred_fan[:,4][None,:,None], boxes_cls_pred_fan[None,:,:4], images.image_sizes)
            # results = self.inference_1(boxes_cls_pred_1[:, 4][None, :, None], boxes_cls_pred_1[None, :, :4],images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results

    def pai_xu(self, list, index):
        sorted_values, sorted_indices = torch.sort(index, descending=True)

        # 根据排序后的索引对tensor_500_5进行重新排序
        list = list[sorted_indices]
        return list

    def f1(self, boxes_cls_pred):
        result = boxes_cls_pred[:-1, 4] - boxes_cls_pred[1:, 4]
        r_max, index = torch.max(result, dim=0)
        return r_max, index

    def res_jie_xi(self, res):
        boxes_cls_pred = torch.cat([res[0].pred_boxes.tensor, torch.unsqueeze(res[0].scores, 1)], 1)

        values = boxes_cls_pred[:, 4]

        # 使用torch.sort()函数对values进行排序，并返回排序后的值和索引
        sorted_values, sorted_indices = torch.sort(values, descending=True)

        # 根据排序后的索引对tensor_500_5进行重新排序
        boxes_cls_pred = boxes_cls_pred[sorted_indices]
        boxes_list = []
        boxes_cls_pred_clone = boxes_cls_pred.detach()
        box_kiou = box_iou_kk(boxes_cls_pred, boxes_cls_pred_clone)
        return boxes_cls_pred, boxes_cls_pred_clone, box_kiou

    def res_jie_xi_1(self, res):
        boxes_cls_pred = torch.cat([res[0].pred_boxes.tensor, torch.unsqueeze(res[0].scores, 1)], 1)

        return boxes_cls_pred

    def forward_1(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = self.inference(box_cls, box_pred, images.image_sizes)
        return results

    def forward_3(self, img,file_name, sx=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if sx:
            img = torch.flip(img, dims=[1])
        else:
            img = torch.flip(img, dims=[2])
        image_dict = {}
        image_dict['file_name'] = str(1)
        image_dict['height'] = img.shape[1]
        image_dict['width'] = img.shape[2]
        image_dict['image'] = img
        image_dict['image_id'] = 1
        batched_inputs = [image_dict]
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]
        ff = []
        for f, p in zip(features, self.back_l):
            ff.append(p(f))
        features = ff
        # draw_features(features,file_name,fan_zhuan=True)
        # draw_features(features, batched_inputs[0]['image'], file_name,fan_zhuan=True)

        # Prepare Proposals.
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = self.inference(box_cls, box_pred, images.image_sizes)
        return results

    def rong_he(self, boxes, boxes_fan):

        pass

    def yuan2fen(self, boxes, scale, xy):
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - xy[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - xy[1]
        boxes = boxes * scale

        return boxes

    def forward_2(self, batched_inputs, boxes):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """

        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]

        # Prepare Proposals.
        proposal_boxes = torch.zeros(500, 4, device=boxes.device)
        proposal_boxes[:, :2] = 0.5
        proposal_boxes[:, 2:] = 1
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes[:len(boxes)] = boxes

        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = self.inference(box_cls, box_pred, images.image_sizes)
        return results

    def prepare_targets(self, targets, ignore):
        new_targets = []
        for targets_per_image, ignore_per_image in zip(targets, ignore):
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["ignore_xyxy"] = ignore_per_image.gt_boxes.tensor.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def inference_1(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal:
            scores = box_cls
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):

        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh

    def segment_image(self, image, boxes):
        segmented_images = []
        zuo_biao = []
        scale = 4

        # 遍历每个预测框
        for box in boxes:
            # 将预测框转换为整数坐标
            x1, y1, x2, y2 = box.int().tolist()
            h, w = y2 - y1, x2 - x1
            h, w = h * 2, w * 2
            x1, y1 = x1 - w / 4, y1 - h / 4
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > image.shape[2]:
                w = image.shape[2] - x1
            if y2 > image.shape[1]:
                h = image.shape[1] - y1
            y1 = int(y1)
            x1 = int(x1)
            h = int(h)
            w = int(w)
            # print(x1,y1)
            # print(w,h)
            zuo_biao.append([x1, y1])
            # 使用crop函数从原始图像中获取感兴趣的区域
            segmented_image = crop(image, y1, x1, h, w)
            if w > 0 and h > 0:
                up_s_i = F.interpolate(segmented_image, scale_factor=scale, mode='nearest')
            segmented_images.append(up_s_i)

        return zuo_biao, segmented_images, scale


def draw_box(path, images_boxes, shape, fanzhuan=False, th=0.5, line_width=1):
    color = ['red', 'yellow', 'green', 'black', 'white', 'orange', 'brown']
    # 读入方框坐标

    boxes = images_boxes
    i = 0
    image = Image.open(path)
    # if fanzhuan:
    #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
    for box in boxes:

        if i == 7:
            i = 0
        [x1, y1, x2, y2] = box[:4]
        x1, x2 = x1 / shape[2] * image.size[0], x2 / shape[2] * image.size[0]
        y1, y2 = y1 / shape[1] * image.size[1], y2 / shape[1] * image.size[1]
        # x2, y2 = int(x1 + width) , int(y1 + height)
        # 绘制方框
        if box[4] < th:
            continue
        draw = ImageDraw.Draw(image)

        draw.rectangle([x1, y1, x2, y2], outline=color[i], width=line_width)
        label = "{:.2%}".format(round(box[4].item(), 4))
        # 保存图像
        draw.text((x1 + 10, y1 + 10), label, fill=color[i])
        i = i + 1

        # 显示图像
        # image.show()

    i_path = '/mnt/Data/HOME/ksn/E2EDET-main/out_cocoperson/' + path.split("/")[-1].split('.')[0] + '_1.jpg'
    if fanzhuan:
        i_path = '/mnt/Data/HOME/ksn/E2EDET-main/out_cocoperson/' + path.split("/")[-1].split('.')[0] + 'f' + '.jpg'
    print(i_path)
    image.save(i_path)



def save_as_img(img, name):
    tensor_to_image = transforms.ToPILImage()
    image = tensor_to_image(img)
    path = '/mnt/Data/HOME/ksn/E2EDET-main/out_cocoperson/' + name + '.jpg'

    # 保存图像到磁盘
    image.save(path)
    return path
class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        # groups=dim 意味着将输入通道分成 dim 个组，然后对每个组执行卷积操作。
        # 它将输入通道分成若干组，然后每组进行卷积操作，最后将结果合并。
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim,dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        self.dcn = build_deformable_conv3()
        # self.conv_pengzhang = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        # self.conv = self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)





        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )


    def forward(self, x):
        identity = x
        # x = self.dcn(x)
        # x = self.norm1(x)
        # x = self.conv1(x)
        #
        # x = self.conv2(x)
        x = self.dcn(x)
        m = min(x.shape[2],x.shape[3])
        if m>9:
            x = torch.cat([self.conv3_19(x), self.conv3_13(x),self.conv3_7(x)], dim=1)
        else:
            x = torch.cat([x, self.conv3_13(x),self.conv3_7(x)], dim=1)
        x = self.mlp(x)

        x = identity + x
        return x


def draw_features(heat,img,file_name,fan_zhuan=False):
    heat0 = heat[0]
    size = tuple([i for i in heat0.shape[:2]])
    heat1 = F.interpolate(heat[1], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat2 = F.interpolate(heat[2], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat3 = F.interpolate(heat[3], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat = (heat0 + heat1 + heat2 + heat3)/4
    if fan_zhuan:
        heat = torch.flip(heat, dims=[3])
    heat = heat.data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除

    heat = heat[0:256, :]  # 切片获取某几个通道的特征图
    heatmap = np.maximum(heat, 0)  # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
    heatmap /= np.max(heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
    # plt.matshow(heatmap)               # 可以通过 plt.matshow 显示热力图
    # plt.show()

    # 用cv2加载原始图像
    img = cv2.imread(file_name)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap) # 将特征图转换为uint8格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
    heat_img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)  # 将伪彩色图与原始图片融合
    # heat_img = heatmap * 0.5 + img 　　　　　　        　　　 # 也可以用这种方式融合
    # cv2.imwrite('./heat_all_3.jpg', heat_img)
    i_path = '/mnt/Data/HOME/ksn/E2EDET-main/heatmap/' + file_name.split("/")[-1].split('.')[0] + 'final.jpg'
    if fan_zhuan:
        i_path = '/mnt/Data/HOME/ksn/E2EDET-main/heatmap/' + file_name.split("/")[-1].split('.')[0] + 'final_f.jpg'

    cv2.imwrite(i_path, heat_img)
    print(i_path)
def draw_features1(heat,img,file_name,fan_zhuan=False):
    heat0 = heat[0]
    size = tuple([i for i in heat0.shape[:2]])
    heat1 = F.interpolate(heat[1], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat2 = F.interpolate(heat[2], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat3 = F.interpolate(heat[3], tuple([i for i in heat0.shape[2:]]), mode='bilinear', align_corners=False)
    heat = (heat0 + heat1 + heat2 + heat3)/4
    if fan_zhuan:
        heat = torch.flip(heat, dims=[3])
    heat = heat.data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    heat = heat[0:256, :]  # 切片获取某几个通道的特征图
    heatmap = np.maximum(heat, 0)  # heatmap与0比较
    heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
    heatmap /= np.max(heatmap)  # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
    # plt.matshow(heatmap)               # 可以通过 plt.matshow 显示热力图
    # plt.show()

    # 用cv2加载原始图像
    img = cv2.imread(file_name)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap) # 将特征图转换为uint8格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
    heat_img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)  # 将伪彩色图与原始图片融合
    # heat_img = heatmap * 0.5 + img 　　　　　　        　　　 # 也可以用这种方式融合
    # cv2.imwrite('./heat_all_3.jpg', heat_img)
    i_path = '/mnt/Data/HOME/ksn/E2EDET-main/heatmap/' + file_name.split("/")[-1].split('.')[0] + 'base.jpg'


    cv2.imwrite(i_path, heat_img)
    print(i_path)
