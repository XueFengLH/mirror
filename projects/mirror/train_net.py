# !/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Sparse-RCNN(github: https://github.com/PeizeSun/SparseR-CNN) created by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os, sys
import os.path as osp
import itertools, time, torch
from typing import Any, Dict, List, Set
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, CityscapesInstanceEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from config import add_sparsercnn_config
from dataset_mapper import SparseRCNNDatasetMapper
from models.detector import SparseRCNN
# 注册数据集
from crowdhuman_eval import ch_eval
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
#声明类别，尽量保持
CLASS_NAMES =['person']
# 数据集路径
DATASET_ROOT = '/mnt/Data/HOME/ksn/E2EDET-main/datasets/coco1'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')

TRAIN_JSON = os.path.join(ANN_ROOT, 'person_keypoints_train2017.json')
#VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = os.path.join(ANN_ROOT, 'person_keypoints_val2017.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "cocoperson_train": (TRAIN_PATH, TRAIN_JSON),
    "cocoperson_val": (VAL_PATH, VAL_JSON),
}

# CLASS_NAMES =['OG', 'F', 'M', 'W', 'T', 'HG']
# # 数据集路径
# DATASET_ROOT = '/mnt/Data/HOME/ksn/E2EDET-main/datasets/ts2023'
# ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
#
# TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
# VAL_PATH = os.path.join(DATASET_ROOT, 'test')
#
# TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
# #VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
# VAL_JSON = os.path.join(ANN_ROOT, 'test.json')
#
# # 声明数据集的子集
# PREDEFINED_SPLITS_DATASET = {
#     "ts_train": (TRAIN_PATH, TRAIN_JSON),
#     "ts_test": (VAL_PATH, VAL_JSON),
# }

#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


# 注册数据集和元数据
def plain_register_dataset():
    #训练集
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="coco_my_val"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        if i == 200:
            break


class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = SparseRCNNDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    # crowdhuman_test
                    try:
                        if self._step_count % 5000 == 19 and self._step_count > 5000:
                            ch_eval(cfg.OUTPUT_DIR, self._step_count)
                    except:
                        pass
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):

    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg['MODEL']['DEVICE'] = 'cpu'
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):

    cfg = setup(args)
    # cfg['MODEL']['DEVICE'] = 'cpu'
    register_dataset()
    if args.eval_only:

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        try:
            original_path = args.opts[1]
            filename, extension = os.path.splitext(original_path)
            original_number = int(filename.split('_')[-1])
            ch_eval(cfg.OUTPUT_DIR, original_number)
        except:
            pass
        res_filename = f"{filename[:-len(str(original_number))]}{'res_coco'}{'.txt'}"
        with open(res_filename, 'a') as file:

            for key, value in res.items():
                file.write(f'{"轮数"}: {round(original_number / 3000)}\n')
                file.write(f'{key}: {value}\n')
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)


    # # 原始路径
    # original_path = args.opts[1]
    #
    # # 提取文件名和扩展名
    # filename, extension = os.path.splitext(original_path)
    #
    # # 循环10次
    # for i in range(60):
    #
    #     # 将原始数字提取出来，并加上3000
    #     original_number = int(filename.split('_')[-1])
    #     new_number = original_number + 3000 * i
    #
    #     # 构建新的文件名
    #     new_filename = f"{filename[:-len(str(new_number))]}{new_number}{extension}"
    #
    #     # 输出新的文件路径
    #     new_path = os.path.join(os.path.dirname(original_path), new_filename)
    #     print(new_path)
    #     args.opts[1] = new_path
    #     launch(
    #         main,
    #         args.num_gpus,
    #         num_machines=args.num_machines,
    #         machine_rank=args.machine_rank,
    #         dist_url=args.dist_url,
    #         args=(args,),
    #     )

    launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
    )
