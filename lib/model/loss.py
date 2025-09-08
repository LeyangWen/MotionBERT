import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
from .SkeletonAngles import VEHS37SkeletonAngles

# Numpy-based errors

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def get_limb_lens(x, args=False):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    if args == False or args.joint_format.upper() == 'H36M':
        limbs_id = [[0, 1], [1, 2], [2, 3],
                    [0, 4], [4, 5], [5, 6],
                    [0, 7], [7, 8], [8, 9], [9, 10],
                    [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16]
                    ]  # wen: idx for h36M
    elif args.joint_format.upper() == 'RTM-24':
        limbs_id = [[22, 3], [22, 4], [4, 6],
                    [3, 5], [5, 7], [6, 8],
                    [7, 16], [8, 19], [21, 9],
                    [21, 10], [9, 11], [10, 12],
                    [11, 13], [12, 14], [21, 22],
                    [22, 23]]
    elif args.joint_format.upper() == 'RTM-37':
        limbs_id = [
            [22, 0],   # C7 to PELVIS
            [0, 3],    # PELVIS to RHIP
            [0, 4],    # PELVIS to LHIP
            [4, 6],    # LHIP to LKNEE
            [3, 5],    # RHIP to RKNEE
            [5, 7],    # RKNEE to RANKLE
            [6, 8],    # LKNEE to LANKLE
            [7, 9],    # RANKLE to RFOOT
            [8, 10],   # LANKLE to LFOOT
            [11, 1],  # RHAND to RWRIST
            [12, 2],  # LHAND to LWRIST         
            [1, 13],  # RWRIST to RELBOW
            [2, 14],  # LWRIST to LELBOW
            [15, 22],  # RSHOULDER to C7
            [16, 22],  # LSHOULDER to C7
            [13, 15],  # RELBOW to RSHOULDER
            [14, 16],  # LELBOW to LSHOULDER
            [22, 23],  # C7 to C7_d
            [23, 24],  # C7_d to SS
            [23, 18],  # C7_d to Thorax
            [24, 18],  # SS to Thorax
            [17, 20],  # HEAD to REAR
            [17, 21],  # HEAD to LEAR
            [17, 19],  # HEAD to HDTP
            [25, 26],  # RAP_b to RAP_f
            [25, 15],  # RAP_b to RSHOULDER
            [26, 15],  # RAP_f to RSHOULDER
            [27, 28],  # LAP_b to LAP_f
            [27, 16],  # LAP_b to LSHOULDER
            [28, 16],  # LAP_f to LSHOULDER
            [29, 30],  # RLE to RME
            [29, 13],  # RLE to RELBOW
            [30, 13],  # RME to RELBOW
            [31, 32],  # LLE to LME
            [31, 14],  # LLE to LELBOW
            [32, 14],  # LME to LELBOW
            [33, 34],  # MCP
            [33, 11], # RMCP to RHAND
            [34, 11], # RMCP to RHAND
            [35, 36],  # MCP
            [35, 12], # LMCP to LHAND
            [36, 12], # LMCP to LHAND
        ]
    elif args.joint_format.upper() == 'HAND-21':
        """
            0: Wrist
            1-4: Thumb
            5-8: Index
            9-12: Middle
            13-16: Ring
            17-20: Pinky
        """
        limbs_id = [[0, 1], [1, 2], [2, 3], [3, 4],
                    [0, 5], [5, 6], [6, 7], [7, 8],
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16],
                    [0, 17], [17, 18], [18, 19], [19, 20]]
    elif args.joint_format.upper() == 'UBHAND-48':
        """
            0: left shoulder
            1: right shoulder
            2: left arm
            3: right arm
            4: left forearm 
            5: righ forearm
            6: left wrist
            7-10: left thumb
            11-14: left index
            15-18: left middle
            19-22: left ring
            23-26: left pinky
            27: right wrist
            28-31: right thumb
            32-35: right index
            36-39: right middle
            40-43: right ring
            44-47: right pinky
        """
        limbs_id = [[0, 2], [2, 4], [4, 6], 
                    [0, 3], [3, 5], [5, 6],
                    [6, 7], [7, 8], [8, 9], [9, 10],
                    [6, 11], [11, 12], [12, 13], [13, 14],
                    [6, 15], [15, 16], [16, 17], [17, 18],
                    [6, 19], [19, 20], [20, 21], [21, 22],
                    [6, 23], [23, 24], [24, 25], [25, 26],
                    [27, 28], [28, 29], [29, 30], [30, 31],
                    [27, 32], [32, 33], [33, 34], [34, 35],
                    [27, 36], [36, 37], [37, 38], [38, 39],
                    [27, 40], [40, 41], [41, 42], [42, 43],
                    [27, 44], [44, 45], [45, 46], [46, 47]
                    ]
    else:
        raise ValueError(f"args.joint_format: {args.joint_format} not recognized")

    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x, args=False):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x, args)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var

def loss_limb_gt(x, gt, args=False):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x, args)
    limb_lens_gt = get_limb_lens(gt, args) # (N, T, 16)
    limb_loss = nn.L1Loss()(limb_lens_x, limb_lens_gt)
    return limb_loss
    # center_loss = loss_center(x, args)
    # # todo: hijacking limb loss for center loss right now, make it separate later
    # return 2* center_loss #+ limb_loss

def loss_center(x, args=False):
    """
    only for VEHS-37 kpts, enforce shoulder center to be in center of back & front of shoulder, same for elbow and wrist
    Input: (N, T, 37, 3), (N, T, 37, 3)
    """
    loss = 0.0
    if args and args.joint_format.upper() == 'RTM-37':
        # Indices in rtm_pose_37_keypoints_vicon_dataset_v1 (zero-based)
        # Centers: RSHOULDER, LSHOULDER, RELBOW, LELBOW, RHAND, LHAND
        center_ids = [15, 16, 13, 14, 11, 12]

        # Supporting pairs:
        # Shoulders: RAP_b/RAP_f, LAP_b/LAP_f
        # Elbows: RLE/RME, LLE/LME
        # Hands: RMCP2/RMCP5, LMCP2/LMCP5
        support_pairs = [
            (25, 26),  # R shoulder supports
            (27, 28),  # L shoulder supports
            (29, 30),  # R elbow supports
            (31, 32),  # L elbow supports
            (33, 34),  # R hand supports
            (35, 36),  # L hand supports
        ]

        # Select predicted and GT points
        # x, gt: (N, T, 37, 3)
        # Gather both support points for all pairs → (N, T, K, 2, 3)
        pred_support_pts = x[:, :, support_pairs, :]    # (N, T, K, 2, 3)
        pred_mid_pts = x[:, :, center_ids, :]          # (N, T, K, 3)

        # Compute midpoint along the "2" axis
        pred_support_mid = pred_support_pts.mean(dim=3) # (N, T, K, 3)

        loss = loss_mpjpe(pred_support_mid, pred_mid_pts)

    return loss


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)

def get_angles(x, args=False):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    if args and args.joint_format.upper() == 'RTM-37':
        # run through SkeletonAngles, eary return
        joint_angles = VEHS37SkeletonAngles(x)
        return joint_angles.get_angles()
    elif args == False or args.joint_format.upper() == 'H36M':
        pass  # run the main code
    else:
        warnings.warn('WARNING: get_angles is set for H36M joint index now, if the angle weight is not 0 and you are using different joint idx, need to modify in loss.py')
    limbs_id = [[0,1], [1,2], [2,3],
        [0,4], [4,5], [5,6],
        [0,7], [7,8], [8,9], [9,10],
        [8,11], [11,12], [12,13],
        [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(x, gt, args=False):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x, args)
    limb_angles_gt = get_angles(gt, args)
    if False: # OG motionbert code
        return nn.L1Loss()(limb_angles_x, limb_angles_gt)
    else: # wen: cos sin dot product loss
        return angle_cos_diff(limb_angles_x, limb_angles_gt)

def loss_angle_velocity(x, gt, args=False):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x, args)
    gt_a = get_angles(gt, args)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    if False: # OG motionbert code
        return nn.L1Loss()(x_av, gt_av)
    else:
        return angle_cos_diff(x_av, gt_av)


def angle_cos_diff(angle1, angle2):
    # angles in radians; any shape
    a1 = angle1.float()
    a2 = angle2.float()
    vx = torch.stack([torch.cos(a1), torch.sin(a1)], dim=-1)
    vg = torch.stack([torch.cos(a2), torch.sin(a2)], dim=-1)
    cos_sim = (vx * vg).sum(dim=-1).clamp(-1.0, 1.0)  # safety clamp
    return (1.0 - cos_sim).mean()