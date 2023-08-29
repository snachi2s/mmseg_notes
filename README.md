# mmsegmentation

Official documentation: [mmseg](https://mmsegmentation.readthedocs.io/en/main/advanced_guides/index.html)

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

[For adding custom components](https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_models.html)

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
  ```python
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
   ```python
   from .greenhouse import GreenHouseDataset
   ```
3. Creating our custom dataset config file `configs/_base_/datasets/greenhouse.py`
   
   ```python
    dataset_type = 'GreenHouseDataset'
    data_root = 'data/GreenHouse'
    ...
   ```
This is the dataloader file for training and testing. 
**NOTE:** In the given example, validation dataset is used for testing

4. Add dataset meta information in `mmseg/utils/class_names.py`
   ```python
   def greenhouse_classes():
      return ["Pipe", "Floor", "Background"]
   
   def greenhouse_pallete():
      return [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255]
      ]
    ```

With this, our custom dataloader is ready for training.

## Creating a config file

- Components that are required to be defined for creating a config file for training are,
    - dataset loader
    - model
    - scheduler
    - default_runtime (for visualization and runtime setting)
All these are defined under `mmseg\configs\_base_`

## Dataset loader
      location: `mmsegmentation\configs\_base_\datasets\greenhouse.py`
- Custom dataset is ready to be loaded and the next step would be creating dataloaders with augmentations
- All the functionalities are defined as type annotations in the mmsegmentation framework, for example, if we want to load images from the path, we use `dict(type='LoadImageFromFile')` which uses the mmseg api's to do the functionality. All the augmentations and dataloaders are defined as type annotations.

```python
dataset_type = 'GreenHouseDataset'
    data_root = 'data/GreenHouse'
    img_scale = (896, 512) #https://github.com/open-mmlab/mmsegmentation/issues/887
    crop_size = (448,448)  
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
In the `train_dataloader`, there is a component called `type='RepeatDataset'`, it repeats the current dataset for `N` times for training. It will be somewhat useful incase of smaller datasets.


## Defining the model
      location: `mmsegmentation/configs/_base_/models/segformer_mit-b0.py`
   - We define the data preprocessor like mean, and std.dev of the dataset, defining encoder and decoder head with number of classes,  
   - In most cases, we will train on the available segmentation models so there won't be many changes in the respective model file. If we want to implement a new model from scratch, then refer [here](https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_models.html)
   - Necessary changes:
       - Changing the number of classes (to be segmented) in the decoder 
       - In our case, `num_classes=4` (with background)
   - Other changes:
       - Trying out with different decoder heads if it is not already implemented in mmsegmentation

## Scheduler
      `location: mmsegmentation/configs/_base_/schedules/schedule_80k.py`
  - Here, we define the optimizer, learning rate scheduler, max. iterations, and checkpoints storage
  - Training configurations can be defined based on iterations or epochs. Validation loops and checkpoints can be performed for every epoch or on a periodic basis.

## default_runtime
      `location: mmsegmentation/configs/_base_/default_runtime.py`
   - In this file, we define the visualization and logger hooks

Once these components are set up, the next would be to define the config file for our model.

## Writing a config file 
    `location: mmsegmentation/configs/model_name.py`
  
  - mmseg has several SOTA models pre-trained under standard datasets. We can adapt those same config files if we have our custom dataset in the standard formats such as Cityscapes, PascalVOC, and ade20k.., Or, we can create custom configs for our dataset
    
  - Every config file will inherit one or many components (datasets, models, scheduler) from the `_base_`. Based on this inheritance and their corresponding object creation contents of the config file can vary. Following are the example cases I encountered after cloning the repository initially
    
    - When we look at the config file for UNet under `mmsegmentation/configs/unet/unet_s5-d16_deeplabv3_4xb4-40k_chase-db1-128x128.py`
      ```python
       _base_ = [
          '../_base_/models/deeplabv3_unet_s5-d16.py',
          '../_base_/datasets/greenhouse.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_40k.py'
      ]
      crop_size = (256,256)
      data_preprocessor = dict(size=crop_size)
      model = dict(
          data_preprocessor=data_preprocessor,
          test_cfg=dict(crop_size=(256,256), stride=(85, 85)))
      ```
      Here, we inherited all four components for the config file, thus we just reloaded the data_preprocessor for resizing.

    - For the config file of PIDnet under `mmsegmentation/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py`
      ```python
            _base_ = [
          '../_base_/datasets/greenhouse.py',
          '../_base_/default_runtime.py'
        ]
  
      class_weight = [1.456, 0.4, 0.4, 0.1] 
      checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
      crop_size = (512,512)
      data_preprocessor = dict(
          type='SegDataPreProcessor',
          mean=[123.675, 116.28, 103.53], #TODO: calculation 
          std=[58.395, 57.12, 57.375],
          bgr_to_rgb=True,
          pad_val=0,
          seg_pad_val=255,
          size=crop_size)
      norm_cfg = dict(type='SyncBN', requires_grad=True)
      model = dict(
          type='EncoderDecoder',
          data_preprocessor=data_preprocessor,
          backbone=dict(
              type='PIDNet',
              in_channels=3,
              channels=32,
              ppm_channels=96,
              num_stem_blocks=2,
              num_branch_blocks=3,
              align_corners=False,
              norm_cfg=norm_cfg,
              act_cfg=dict(type='ReLU', inplace=True),
              init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
          decode_head=dict(
              type='PIDHead',
              in_channels=128,
              channels=128,
              num_classes=4,
              norm_cfg=norm_cfg,
              act_cfg=dict(type='ReLU', inplace=True),
              align_corners=False,  #if True set the img_scale to odd (nx+1) pixels
              loss_decode=[
                  dict(
                      type='TverskyLoss',
                      ignore_index=255,
                      class_weight=class_weight,
                      loss_weight=0.4),
                  dict(
                      type='OhemCrossEntropy',
                      thres=0.9,
                      min_kept=131072,
                      class_weight=class_weight,
                      loss_weight=1.0),
                  dict(type='BoundaryLoss', loss_weight=20.0),
                  dict(
                      type='OhemCrossEntropy',
                      thres=0.9,
                      min_kept=131072,
                      class_weight=class_weight,
                      loss_weight=1.0)
              ]),
          train_cfg=dict(),
          test_cfg=dict(mode='whole'))
      
      iters = 120000
      # optimizer
      optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
      optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
      # learning policy
      param_scheduler = [
          dict(
              type='PolyLR',
              eta_min=0,
              power=0.9,
              begin=0,
              end=iters,
              by_epoch=False)
      ]
      # training schedule for 120k
      train_cfg = dict(
          type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
      val_cfg = dict(type='ValLoop')
      test_cfg = dict(type='TestLoop')
      default_hooks = dict(
          timer=dict(type='IterTimerHook'),
          logger=dict(type='LoggerHook', interval=12000, log_metric_by_epoch=False),
          param_scheduler=dict(type='ParamSchedulerHook'),
          checkpoint=dict(
              type='CheckpointHook', by_epoch=False, interval=iters // 10),
          sampler_seed=dict(type='DistSamplerSeedHook'),
          visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
      
      randomness = dict(seed=304)
      ```
      Here, only runtime and dataloaders are inherited from `_base_`, so the `scheduler` and `model` are defined(need to be) inside this config file. 

## Training 

- Config file is done and the next step would be the training
For training,
`python tools/train.py configs/model/model_with_desired_configuration.py`

- train.py is located under `mmsegmentation/tools`, and it needs config file as an argument. For example, training our GreenHouse dataset with the given pidnet-s config file will be
    `python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py`

- For understanding the meaning behind the naming of config files in mmseg, refer [here](https://mmsegmentation.readthedocs.io/en/main/user_guides/1_config.html#:~:text=for%20detailed%20documentation.-,Config%20Name%20Style,-We%20follow%20the)


## Visualization

### Training visualization

- mmseg also supports tensorboard, for setting up the visualization hook follow the instructions
  - Dependancies
    ```
    pip install tensorboardX
    pip install future tensorboard
    ```
  - Set up the visualization hook for using tensorboard functionalities by adding/modifying the following lines in `mmsegmentation/configs/_base_/default_runtime.py`
    ```
    vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
    visualizer = dict(
        type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
    ```
    
  - Set up the scheduler for the changed logger, by adding/modifying the corresponding scheduler  `_base_/schedules/schedule_120k.py`
    ```
    default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=2000))
    ```
    We set the `draw` parameter of the visualization hook to store the predictions. By setting the draw parameter, mmseg's visualization hook draws the predicted segmentation masks on the image and stores it in the directory we input in the command line(`work_dir/visualization/vis_image`)
  
- For visualizing the predictions during training, we need to pass in the argument for `--work-dir` which expects a folder name as input. By default, the predictions are not stored, so it is essential to pass in the directory while invoking the training script. Example command will look like this, 
  `python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py --work-dir work_dir/visualization`

- For testing,
  `python tools/train.py configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py --work-dir work_dir/visualization`


# Reference
- https://github.com/open-mmlab/mmsegmentation



