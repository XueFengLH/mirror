#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install and build libs
```
git clone https://github.com/megvii-model/Iter-E2EDET.git
cd Iter-E2EDET
python3 setup.py build develop
```

2. Load the CrowdHuman images from [here](https://www.crowdhuman.org/download.html) and its annotations from [here](https://drive.google.com/file/d/11TKQWUNDf63FbjLHU9iEASm2nE7exgF8/view?usp=sharing). Then update the directory path of the CrowdHuman dataset in the config.py.
```
cd projects/mirror
vim config.py
imgDir = 'CrowdHuman/images'
json_dir = 'CrowdHuman/annotations'
```

3. Train Iter SparseR-CNN
```
cd projects/mirror
python3 train_net.py --num-gpus 8 \
    --config-file configs/50e.6h.500pro.ignore.yaml

```

4. Evaluate Iter SparseR-CNN. You can download the pre-trained model from [here](https://drive.google.com/file/d/1LTP-Qfe6QsnhCOL3e-lxuuXEqfJ55sgj/view?usp=sharing) for direct evaluation.
```
-- python3 train_net.py --num-gpus 8 \
    --config-file configs/50e.6h.500pro.ignore.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

