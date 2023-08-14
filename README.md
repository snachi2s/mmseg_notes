# mmsegmentation

## Requirements

- Basic installation 

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

## Installing mmsegmentation

```
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Dataset structure for using our custom dataset 

- Default dataset path: **$MMSEGMENTATION/data**

```
- data
  - GreenHouseDataset
    - images
      - training
        - .....png
      - validation
        - .....png
    - annotations
      - training
        - ....png
      - validation
        - ....png

```

## Pipeline and what needs to be changed?

`python tools/train.py configs/model/model_with_desired_configuration.py`

- train.py is located under `mmsegmentation/tools`, and it needs config file as an argument
- mmseg has several SOTA models pre-trained under standard datasets like Cityscapes, PascalVOC, ADE20k..,we can adapt those same config files to train and test our custom dataset. 
- 






