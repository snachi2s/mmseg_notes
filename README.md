# mmsegmentation

## Requirements

## Creating conda environment (Recommended)
```
conda create --name mmsegmentation python=3.8 -y
conda activate mmsegmentation
```

## Pytorch environment
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Dependancies

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmdet>=3.0.0rc4" 
```

## Setting up mmsegmentation workspace

```
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```
**NOTE:** For installing cpu only version refer [here](https://mmsegmentation.readthedocs.io/en/main/get_started.html#:~:text=%2Dc%20pytorch-,On%20CPU%20platforms%3A,-conda%20install%20pytorch) 

[adding custom components](https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_models.html)

## Dataset structure for using our custom dataset 

- Default dataset path: **$MMSEGMENTATION/data**

```
├── data
│   ├── Greenhouse
│     ├── images
│         ├── training 
│             ├── image_1.png
│             ├── ....
│             ├── image_n.png
│         ├── validation 
│             ├── image_1.png
│             ├── ....
│             ├── image_n.png
│     ├── annotations
│         ├── training 
│             ├── mask_1.png
│             ├── ....
│             ├── mask_n.png
│         ├── validation 
│             ├── mask_1.png
│             ├── ....
│             ├── mask_n.png

```

## Creating dataloader for our custom dataset

### Adding new datasets

For official documentation, refer [here](https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html)

1. Create a new dataset file `mmseg/datasets/greenhouse.py`
  ```
  from mmseg.registry import DATASETS #register the dataset
  from .basesegdataset import BaseSegDataset
  import mmengine.fileio as fileio
  
  @DATASETS.register_module()
  class GreenHouseDataset(BaseSegDataset):
  
      METAINFO = dict(
          classes=("Pipe", "Floor", "Background"),
          palette=[[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255]]) 
  
      def __init__(self,  img_suffix=".png", 
                   seg_map_suffix=".png",
                   reduce_zero_label=True, #used when background represents 0 in our annotations
                   **kwargs) -> None:
          super().__init__(img_suffix=img_suffix,
                           seg_map_suffix=seg_map_suffix,
                           reduce_zero_label=reduce_zero_label,
                           **kwargs)
          assert fileio.exists(
              self.data_prefix['img_path'], backend_args=self.backend_args)
  ```
Basically, we are creating a custom class for our dataset and adding metadata such as class names and palette for better visualization. One other thing to note here is the **img_suffix** and **seg_map_suffix** which in our case is just **.png**. To suppress the background label while training/visualization we set the **reduce_zero_label** here.

- **@DATASETS.register_module()** --> mmseg needs to register each of the datasets with a unique class name
  
2. Import our module in `mmseg/datasets/__init__.py`
   `
   from .greenhouse import GreenHouseDataset
   `
3. Creating our custom dataset config file `configs/_base_/datasets/greenhouse.py`
   
   ```
    dataset_type = 'GreenHouseDataset'
    data_root = 'data/GreenHouse'
    img_scale = (896, 512) #https://github.com/open-mmlab/mmsegmentation/issues/887
    crop_size = (449,449)  #during training
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='RandomResize',
            scale=img_scale, #actual dimensions
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #cat_max_ratio: The max area ratio that could be occupied by single category.
        dict(type='RandomFlip', prob=0.5),
        dict(type='GenerateEdge', edge_width=4),
        dict(type='PackSegInputs')
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=img_scale, keep_ratio=True),
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]
    img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    tta_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(
            type='TestTimeAug',
            transforms=[
                [
                    dict(type='Resize', scale_factor=r, keep_ratio=True)
                    for r in img_ratios
                ],
                [
                    dict(type='RandomFlip', prob=0., direction='horizontal'),
                    dict(type='RandomFlip', prob=1., direction='horizontal')
                ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
            ])
    ]
    
    train_dataloader = dict(
        batch_size=2,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type='RepeatDataset',
            times=4000,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                data_prefix=dict(
                    img_path='images/training',
                    seg_map_path='annotations/training'),
                pipeline=train_pipeline)))
    
    val_dataloader = dict(
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/validation',
                seg_map_path='annotations/validation'),
            pipeline=test_pipeline))
    test_dataloader = val_dataloader
    
    val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
    test_evaluator = val_evaluator
  
    #https://mmsegmentation.readthedocs.io/en/main/user_guides/1_config.html
   ```
This is the dataloader file for training and testing. All the functionalities are defined as type annotations in mmsegmentation framework, for example, if we want to load images from the path, we use `dict(type='LoadImageFromFile')` which uses the mmseg api's to do the functionality. All the augmentations, dataloaders are defined as type annotations.

## Pipeline and what needs to be changed?

`python tools/train.py configs/model/model_with_desired_configuration.py`

- train.py is located under `mmsegmentation/tools`, and it needs config file as an argument
- mmseg has several SOTA models pre-trained under standard datasets. We can adapt those same config files if we have our custom dataset in the standard formats such as Cityscapes, PascalVOC, and ade20k.., Or, we can create custom configs for our dataset





