# Leyang notes


## All hardcoded instances of 17 keypoints
- [x] When `arg.flip` is set to `True`, left and right keypoints idx in `lib\utils\utils_data.py` - `flip_data`
- [x] When `loss limb` is used, the limb index in `lib\model\loss.py` - `get_limb_lens(x)`
- [ ] When `angle loss` is used, the limb index in `lib\model\loss.py` - `get_angles(x)`
- [x] `args.args.rootrel`, both true and false, change `train.py` two instances
- [x] Root-relative Errors - calculating `batch_gt` in `train.py`
- [x] In `train.py`, `evaluation` function calculating `root_relative error`, multiple 0 index as pelvis
- [x] In data loader and data reader, keypoint number (17) and camera resolution hardcoded

## Other common mistakes
- [x] Make sure the root_idx is set correctly when generating input dataset in `Vicon-read`, root-z should be 0
- [x] Make sure the file name, camera, activity sequence is the same sequence in the 2D and 3D. 

## Tasks
- [ ] Train both config 2 and 6
- [ ] test on industry video
- [x] show train and validation loss
- [x] find out what to set x, y when conf is 0, nan?
  - In H36M-MB pkl example, min confidence is 0.004, x, y is still normal pixel value. 
- [ ] train on both fully processed & raw 2d