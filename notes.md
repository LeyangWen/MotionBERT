# Leyang notes

## Change config
add `joint_format: Hand-21` and `root_idx: 0`
## All hardcoded instances of 17 keypoints
- [x] When `arg.flip` is set to `True`, left and right keypoints idx in `lib\utils\utils_data.py` - `flip_data`
- [x] When `loss limb` is used, the limb index in `lib\model\loss.py` - `get_limb_lens(x)`
- [ ] When `angle loss` is used, the limb index in `lib\model\loss.py` - `get_angles(x)`
- [ ] `loss_mesh.py`, root idx
- [x] `args.args.rootrel`, both true and false, change `train.py` two instances
- [x] Root-relative Errors - calculating `batch_gt` in `train.py`
- [x] In `train.py`, `evaluation` function calculating `root_relative error`, multiple 0 index as pelvis
- [x] In data loader and data reader, keypoint number (17) and camera resolution hardcoded
- [x] set `args.discard_last_layer`
## Other common mistakes
- [x] Make sure the root_idx is set correctly when generating input dataset in `Vicon-read`, root-z should be 0
- [x] Make sure the file name, camera, activity sequence is the same sequence in the 2D and 3D. 

## Tasks
- [x] Train both config 2 and 6
- [x] test on industry video
- [x] show train and validation loss
- [x] find out what to set x, y when conf is 0, nan?
  - In H36M-MB pkl example, min confidence is 0.004, x, y is still normal pixel value. 
- [x] train on both fully processed & raw 2d

## 3D-6D-SMPL compare

```mermaid
graph LR
A[input gt 2d 24 kpts]
B[input gt 2d 66 kpts]
C[3d 24 kpts]
D[3d 66 kpts]
E[3d SMPL mesh]
H[vicon gt 3d 66 kpts]
A -->|3d model| C
B -->|6d model| D
B -->|mesh model| E

F[Partial ergo angles]
subgraph G[Full ergo angles compare]
    F
end
C --> F
D --> G
E -->|+vid| G
H --> G

I[3DSSPP result compare]
C -->|+assumptions| I
D --> I
E -->|+vid| I
H --> I
```


## Mesh notes
Goal: format VEHS-7M mesh dataset and train on our own, get 66 GT 2D keypoints --> SMPL
- [x] Too big for mesh, downsample by 5
- [x] Motionbert takes SMPL, not SMPLX, need to convert
- [x] MotionBert take netual gender, need to convert
  - [ ] or train gender specific parameter `train_mesh.py` line 70 `smpl = SMPL(args.data_root, batch_size=1).cuda()`, change data_root to male/female pkl, add gender parameter
- [ ] Hard coded places for 17 h36m
  - [x] listed in todos
    - [ ] line 71 `smpl.J_regressor_h36m` make it not about h36m
    - [x] loss_mesh.py root idx `preds_local = preds['kp_3d'] - preds['kp_3d'][:, :, 0:1,:]  # (N, T, 17, 3)` 
  - [ ] For mesh train, only need to change meshLoss and compute_error, rest is for evaluate.

- Test using 17 kpts, Train and infer works
- 
### Convert to 66 kpts input
- Data flow:
  - config MB_pkl file --> [`MotionSMPL` class](lib/data/dataset_mesh.py) --> `SMPLDataset` class init  --> `DataReaderVEHSR3`
    - add condition if dataset=="name", load using `DataReaderVEHSR3`, split into clips here
      - specify dataset name in `train_mesh.py` 
    - `MotionSMPL` class set `J_regressor`, which is from `self.smpl.J_regressor_h36m`, which set [here](lib/utils/utils_smpl.py)
  - 2D data input: def `read_2d` -->  `motion_2d`
  - 3D data input: `motion_smpl_3d`=dict: pose: `motion_smpl_pose` (n, 72), shape: `motion_smpl_shape`(n, 10) --> `motion_verts` (n, 6890, 3) --> *1000 to mm --> `J_regressor` --> `motion_3d_reg` (n, 17 or 66, 3)
  - `motion_smpl_3d` overwrite to 
    ```python
    motion_smpl_3d = {
        'theta': motion_theta,       # smpl pose and shape
        'kp_3d': motion_3d_reg,      # 3D keypoints
        'verts': motion_verts,       # 3D mesh vertices
    }
    ```
  - Loss: `motion_smpl_3d` --> `batch_input`, `batch_gt` in def `train_epoch` 
    - `batch_input` --> `model` --> `output
    - `batch_gt` --> `criterion`
- Step 1: make our own J_regressor, 66 kpts
  - Checking `data/mesh/J_regressor_h36m_correct.npy`, shape (17, 6890), max 0.53, mostly zeros
  - Assuming J_regressor.dot(verts) = 3D keypoints, need to make a new J_regressor for 66 kpts
    - I got smpl vert to 47 surface kpts in soma github, lets put in `vicon-read` - `conversion_scripts/make_J_regressor_VEHS7M_66kpt.py`
    - Done, coped to `data/mesh/J_regressor_VEHS7M_66kpt.npy`
    - Add regressor in (`SMPL` class)[lib/utils/utils_smpl.py]
    - Change `J_regressor` in (`MotionSMPL` class)[lib/data/dataset_mesh.py]
    - Change `J_regressor` in (`train_mesh.py`--> `MeshRegressor` class --> `SMPLRegressor` class)[lib/model/model_mesh.py]
    - **Also need to change `MeshRegressor` in infer wild** not done yet <-------------------------
- Also need to set root index, pelvis is 57 in V2 66-kpts sequence
- Test training
  - Error: Finished all too quick without training

### 20240921
- env setup on mac, use 3.8 python, replace chumpy `pip install git+https://github.com/mattloper/chumpy`


## Hand notes
### 20240920
- Prepared new dataset, all flipped to right hand
- Set flip to false, rootrel to true


