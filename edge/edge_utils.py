import numpy as np
import warnings
import json
from lib.utils.tools import *


def model_pos_coreml(coreml_model, model_input):
    """
    Run coreml model for N (batch) clips (each 243 frame) model input, simulate pytorch model_pos output format
    model_input: Batch x clip (243) x joint x 3, where last one is confidence
    """
    results_clip = []
    assert model_input.shape[-1] == 3  # norm_x, norm_y, confidence
    if len(model_input.shape) == 3:  # if no batch dim
        model_input = model_input.reshape(1, *model_input.shape)
    for idx, in_one in enumerate(model_input):
        one_frame_shape = in_one.shape
        input_data = {
            'input': in_one.reshape([1, *one_frame_shape])
        }
        coreml_out = coreml_model.predict(input_data)['linear_87'].reshape(one_frame_shape)
        results_clip.append(coreml_out)
    return np.array(results_clip)


# todo: make datareader class? if need iterations
def mock_input_pkl(args, save_to_json=False):
    """
    For python-coreml testing, read main pkl file from config file, pkl in h36m-MB format
    Only output args.test_set_keyword, can be train, validate, or test
    output: testset: Frames (first vid from first camera) x J x 3
    """
    pkl_filename = '%s/%s' % ('data/motion3d', args.dt_file)
    dt_dataset = read_pkl(pkl_filename)
    testset = dt_dataset[args.test_set_keyword]['joint_2d'][::args.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
    if not args.no_conf:
        if 'confidence' in dt_dataset[args.test_set_keyword].keys():
            test_confidence = dt_dataset[args.test_set_keyword]['confidence'][::args.sample_stride].astype(np.float32)
            if len(test_confidence.shape) == 2:  # (1559752, 17)
                test_confidence = test_confidence[:, :, None]
        else:
            # No conf provided, fill with 1.
            test_confidence = np.ones(testset.shape)[:, :, 0:1]
        testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]

    # extract first vid only
    camera_names = dt_dataset[args.test_set_keyword]['camera_name']
    first_camera_name = camera_names[0]  # Store the first camera name
    # Assume the last frame is the last one in the list initially
    last_vid_frame = len(camera_names) - 1
    for frame_index, camera_name in enumerate(camera_names):
        if camera_name != first_camera_name:
            last_vid_frame = frame_index - 1  # Update last frame index to the index before the change
            break
    testset = testset[:last_vid_frame]

    if save_to_json:
        json_filename = pkl_filename.replace('.pkl', '.json')
        with open(json_filename, 'w') as f:
            json.dump(testset.tolist(), f)
        print(f"Saved as json to {json_filename}")
    return testset


def normalize_2d(testset, res_h, res_w):
    """
    map to [-1, 1]
    testset: Frames x J x 2 or FxJx3
    """
    testset[:, :, :2] = testset[:, :, :2] / res_w * 2 - np.array([1, res_h / res_w])
    return testset


def denormalize(test_data, res_h, res_w):
    """
    data: (batch N x n_frames(243), 51) or data: (batch N x n_frames, 17, 3)
    batch N x n_frames == total frames
    """
    data = test_data.reshape([-1, 17, 3])
    # denormalize (x,y,z) coordinates for results
    data[:, :, :2] = (data[:, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
    data[:, :, 2:] = data[:, :, 2:] * res_w / 2
    return data # [frame, 17, 3]


def split_infer_clips(test_data, n_frames=243, residual_mode="fill"):
    '''
    Args:
        test_data: Frames x Joints x Dim
        n_frames: default 243, clip length
        residual_mode: discard, fill (with last frame), backfill (test_data[:243])

    Returns:

    '''
    frames, J, D = test_data.shape
    assert frames>n_frames, f"Make sure you have enough frames (current: {frames}) to fill one clip of {n_frames}"
    assert D == 3, "Make sure input dim is 3, with norm_x, norm_y, and confidence"

    # Calculate the number of full clips and the number of frames to keep
    num_full_clips = frames // n_frames
    frames_to_keep = num_full_clips * n_frames
    remaining_frames = frames % n_frames

    if remaining_frames == 0:
        clipped_data = test_data
        keep_idx = slice(0, frames)
    else:
        if residual_mode == 'discard':
            # Discard remaining frames
            clipped_data = test_data[:frames_to_keep]
            keep_idx = slice(0, frames_to_keep)
        elif residual_mode == 'fill':
            clipped_data = np.concatenate((
                                            test_data,
                                            np.repeat(test_data[[-1], :, :], n_frames - remaining_frames, axis=0)
                                           ), axis=0)
            keep_idx = slice(0, frames)
        elif residual_mode == 'backfill':
            # raise NotImplementedError
            backfill_data = test_data[-n_frames:]
            clipped_data = np.concatenate((
                                                test_data[:frames_to_keep],
                                                backfill_data
                                                ), axis=0)
            keep_idx = np.concatenate([
                np.arange(0, frames_to_keep),
                np.arange(n_frames-remaining_frames+frames_to_keep, clipped_data.shape[0])
            ])
    # Reshape the data
    num_clips = clipped_data.shape[0] // n_frames
    output_data = clipped_data.reshape(num_clips, n_frames, J, 3)
    info = {
        'mode': residual_mode,
        'keep_idx': keep_idx
    }
    return output_data, info



