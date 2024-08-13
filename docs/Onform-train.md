# Onform train scripts

## Load environment
```bash
conda activate motionbert
```

## Load dataset
```bash
gcloud auth login

gsutil cp "gs://csegolfdata2024/Onform Test/Data/pose_downsample-1-2-3-4-8_keep1_syn_noNan_noPitch.pkl" data/motion3d/

```

## Dataset preparation
```bash
python -u tools/convert_onform.py \
--dt_root 'data/motion3d/' \
--dt_file 'pose_downsample-1-2-3-4-8_keep1_syn_noNan_noPitch.pkl' \
--test_set_keyword 'test' \
--root_path 'data/motion3d/onform_golf_2/try_0' 
```

## Train

Remove old checkpoint
```bash
rm -rf checkpoint/pose3d/onform_golf_1
```

Train
```bash
nohup python train.py \
--config configs/pose3d/onform_exp/MB_train_golf_fpsAug_syn.yaml \
--test_set_keyword test \
--wandb_project "MotionBert_train_onform" \
--wandb_name "Train_3-noise-TSFilter-synthetic-fpsAug-noPitch" \
--checkpoint checkpoint/pose3d/onform_golf_2 \
--note "synthetic data" \
--selection latest_epoch.bin
```

## Evaluation
```bash
python -u train.py \
--config configs/pose3d/onform_exp/MB_train_golf_fpsAug.yaml \
--wandb_project "MotionBert_eval_onform" \
--wandb_name "Train_3-noise-TSFilter-synthetic-fpsAug-noPitch" \
--note "43 epoch" \
--out_path "experiment/Onform_golf_2/Train_3-noise-TSFilter-synthetic-fpsAug-noPitch" \
--test_set_keyword test \
--evaluate "checkpoint/pose3d/onform_golf_2/latest_epoch.bin" \

```

## Run inference

### Infer with train data loader
First make MB-pkl in the same format as the train data loader and put in data folder.

Then data preprocesse
```bash
python -u tools/convert_inference.py \
--dt_root 'data/motion3d/' \
--dt_file 'MB_infer_input.pkl' \
--test_set_keyword 'test' \
--root_path 'data/motion3d/nateIMU_golf/test_1' \
--res_w 1920 \
--res_h 1920 \

```

```bash
python -u infer3d_train.py \
--config configs/pose3d/onform_exp/MB_infer_nateMU_golf.yaml \
--wandb_project "MotionBert_eval" \
--wandb_name "VIT_input_MB_leyang_V2_inference_nateIMU_golf" \
--note "model == Train_2-noise-TSFilter-synthetic-fpsAug" \
--out_path "experiment/nateIMU_golf/Train_2-noise-TSFilter-synthetic-fpsAug" \
--test_set_keyword test \
--evaluate "checkpoint/pose3d/onform_golf_1/latest_epoch.bin" \
--res_w 1920 \
--res_h 1920 \

```
(Set res_wh to the bigger side of the image)

## todo
- [ ] make new tools/convert_onform.py
- [ ] make new data reader for onform
