# Onform train scripts

## Load environment
```bash
conda activate motionbert
```

## Load dataset
```bash
gcloud auth login
gsutil cp "gs://csegolfdata2024/Onform Test/Data/pose_downsample1_keep1_filt.pkl" pose_downsample1_keep1_filt.pkl
```

## Dataset preparation
```bash
python -u tools/convert_onform.py \
--dt_root 'data/motion3d/' \
--dt_file 'pose_downsample1_keep1_filt.pkl' \
--test_set_keyword 'test' \
--root_path 'data/motion3d/onform_golf_0/lab_1' 
```

## Train

```bash
python train.py \
--config configs/pose3d/onform_exp/MB_train_golf.yaml \
--test_set_keyword test \
--wandb_project "MotionBert_train_onform" \
--wandb_name "lab_1" \
--checkpoint checkpoint/pose3d/onform_golf \
--selection latest_epoch.bin
```


## todo
- [ ] make new tools/convert_onform.py
- [ ] make new data reader for onform
