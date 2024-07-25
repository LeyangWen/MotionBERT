# Onform train scripts

## Load environment
```bash
conda activate motionbert
```

## Load dataset
```bash
gcloud auth login

gsutil cp "gs://csegolfdata2024/Onform Test/Data/pose_downsample-1-2-3-4-8_keep1_filt_noNan.pkl" data/motion3d/

```

## Dataset preparation
```bash
python -u tools/convert_onform.py \
--dt_root 'data/motion3d/' \
--dt_file 'pose_downsample-1-2-3-4-8_keep1_filt_noNan.pkl' \
--test_set_keyword 'test' \
--root_path 'data/motion3d/onform_golf_0/lab_2' 
```

## Train

```bash
rm -rf checkpoint/pose3d/onform_golf
```

```bash
nohup python train.py \
--config configs/pose3d/onform_exp/MB_train_golf_fpsAug.yaml \
--test_set_keyword test \
--wandb_project "MotionBert_train_onform" \
--wandb_name "Train_1-noise-TSFilter-1kpxSquareImg-fpsAug" \
--checkpoint checkpoint/pose3d/onform_golf \
--selection latest_epoch.bin
```


## todo
- [ ] make new tools/convert_onform.py
- [ ] make new data reader for onform
