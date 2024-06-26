# 3D Human Pose Estimation

## Data

1. Download the finetuned Stacked Hourglass detections and our preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to `data/motion3d`.

  > Note that the preprocessed data is only intended for reproducing our results more easily. If you want to use the dataset, please register to the [Human3.6m website](http://vision.imar.ro/human3.6m/) and download the dataset in its original format. Please refer to [LCN](https://github.com/CHUNYUWANG/lcn-pose#data) for how we prepare the H3.6M data.

2. Slice the motion clips (len=243, stride=81)

   ```bash
   python tools/convert_h36m.py
   ```

## Running

**Train from scratch:**

```bash
python train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--checkpoint checkpoint/pose3d/MB_train_h36m
```

**Finetune from pretrained MotionBERT:**

```bash
python train.py \
--config configs/pose3d/MB_ft_h36m.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/pose3d/FT_MB_release_MB_ft_h36m
```

**Evaluate:**

```bash
python train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--evaluate checkpoint/pose3d/MB_train_h36m/best_epoch.bin         
```

# Leyang-VEHS-R3

## Data

1. Process using Vicon-Read/caculateSkeleton.py
2. Slice the motion clips (len=243, stride=81)

   ```bash
   python tools/convert_VEHSR3.py `
   --dt_root 'W:\VEHS\VEHS data collection round 3\processed' `
   --dt_file 'VEHS_3D_downsample5_keep1.pkl' `
   --root_path 'data/motion3d/MB3D_VEHS_R3/3DPose'
   
   # 3D Pose
   ```
   ```bash
   python tools/convert_VEHSR3.py `
   --dt_root 'W:\VEHS\VEHS data collection round 3\processed' `
   --dt_file 'VEHS_6D_downsample5_keep1.pkl' `
   --root_path 'data/motion3d/MB3D_VEHS_R3/6DPose'
   # 6D Pose
   ```
3. copy pkl file to data/motion3d/MB3D_VEHS_R3/3DPose
   ```bash
   copy-item -path "W:\VEHS\VEHS data collection round 3\processed\VEHS_3D_downsample5_keep1.pkl" -destination "data/motion3d/MB3D_VEHS_R3/3DPose"
   copy-item -path "W:\VEHS\VEHS data collection round 3\processed\VEHS_6D_downsample5_keep1.pkl" -destination "data/motion3d/MB3D_VEHS_R3/6DPose"
   ```
## Running



**Finetune from pretrained MotionBERT:**

```bash
python train.py ^
--config configs/pose3d/MB_ft_VEHSR3_3DPose.yaml ^
--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m ^
--checkpoint checkpoint/pose3d/3DPose_VEHSR3 ^
--selection best_epoch.bin ^
--resume checkpoint/pose3d/3DPose_VEHSR3/epoch_7.bin
# 3D Pose
```

```bash
python train.py `
--config configs/pose3d/MB_ft_VEHSR3_6DPose.yaml `
--pretrained checkpoint/pose3d/FT_MB_release_MB_ft_h36m `
--checkpoint checkpoint/pose3d/6DPose_VEHSR3 `
--selection best_epoch.bin
#--resume checkpoint/pose3d/6DPose_VEHSR3/epoch_1.bin `
# 6D Pose
```

Visualize the training process:
```bash
tensorboard --logdir  checkpoint/pose3d/6DPose_VEHSR3/logs
```

**Evaluate:**

```bash
python train.py --config configs/pose3d/MB_train_VEHSR3.yaml --evaluate checkpoint/pose3d/MB_train_VEHSR3_3DPose/best_epoch.bin      
```

```bash
python train.py --config configs/pose3d/MB_train_h36m.yaml --evaluate checkpoint/pose3d/MB_train_h36m/best_epoch.bin      
```

# Note about 2.5d factor

* [1]
* [2]
* ```data['test']['joints_2.5d_image']/data['test']['joint3d_image'] == 2.5factor```
* 
## Wen Note:

- If gt_2d is set in config, the input 2D use the first two dimensions of the 3D pose [here](lib/data/dataset_motion_3d.py)
- If MotionBert's prediction is in camera coord / px. Multiply by 2.5d factor to get meters and compare during test time. 
- Pelvic centered:
  - joint_2d is detection results in px
  - joint3d_image = [gt_joint2d, depth] in px
  - joint_2.5d_image = joint3d_image * 2.5d_factor (normally 4.xx)
- joint_3d_camera = joint_2.5d_image, but camera is origin center

[1]:https://github.com/Walter0807/MotionBERT/issues/21
[2]:https://github.com/Walter0807/MotionBERT/issues/65






