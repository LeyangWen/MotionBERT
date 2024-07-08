import coremltools as ct
import torch
import pickle
import numpy as np
import argparse

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.model.loss import *

## Need to run in base dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m.yaml", help="Path to the config file.")
    parser.add_argument('--example_in_output', default='edge/pytorch_output/input_output0.pkl', type=str)
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-o', '--output_coreml_file', type=str, default=r'edge/MB_h36m.mlpackage')
    parser.add_argument('--compare_output', default=True, type=bool)
    parser.add_argument('--batch_input', default=True, type=bool)

    opts = parser.parse_args()
    return opts


def load_checkpoint(chk_filename, model, map_location='cpu', verbose=False):
    # Load the checkpoint
    checkpoint = torch.load(chk_filename, map_location=map_location)
    # Remove 'module.' prefix if present (for DataParallel compatibility)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}
    if verbose:
        print("@"*20, 'Dict keys', "@"*20)
        for key in new_state_dict.keys():
            if 'blocks_st' in key:
                print("_"*10, key)
            elif 'blocks_ts' in key:
                print("*" * 10, key)
            else:
                print(key)
        print("@"*40)
    # Load the state dict into your model
    model.load_state_dict(new_state_dict, strict=True)

def load_in_output(input_output_file):
    # Load the input/output shapes
    with open(input_output_file, 'rb') as f:
        input_output = pickle.load(f)
    input_tensor = input_output['input'].to(dtype=torch.float32)
    gt_output_tensor = input_output['gt'].to(dtype=torch.float32)
    pytorch_output_tensor = input_output['output'].to(dtype=torch.float32)
    return input_tensor, gt_output_tensor, pytorch_output_tensor

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    # Step 0.5: Load input output examples
    input, gt_output, pytorch_output = load_in_output(opts.example_in_output)
    input_example = input[0:1]

    # Step 1: Load model and weights
    # args.backbone = 'DSTformer_coreml'  # modified backbone
    model_backbone = load_backbone(args)
    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)
    load_checkpoint(chk_filename, model_backbone)
    model_backbone.eval()

    # Step 2: Trace model
    print("Tracing model")
    traced_model = torch.jit.trace(model_backbone, input_example)

    # Step 3: Convert & Save
    print("Converting model")
    # # Convert the model
    # print(input_example.shape)  # torch.Size([1, 243, 17, 3])
    if opts.batch_input:
        input_ct = ct.TensorType(shape=(ct.RangeDim(1, 128), 243, 17, 3), name="input")
    else:  # single input torch.Size([1, 243, 17, 3])
        input_ct = ct.TensorType(name='input', shape=input_example.shape)
    mlmodel = ct.convert(traced_model,
                         inputs=[input_ct],
                         source="pytorch",
                         convert_to="mlprogram"
                         )
    print(f"Saving model to {opts.output_coreml_file}")
    # Save the converted model
    mlmodel.save(opts.output_coreml_file)

    # Step 4: Compare pytorch vs coreml-python output
    if opts.compare_output:
        coreml_model = ct.models.MLModel(opts.output_coreml_file)
        mpjpe_compare = np.ones([0])
        mpjpe_pytorch = np.ones([0])
        mpjpe_coreml = np.ones([0])
        for idx, (in_one, gt_out, pytorch_out) in enumerate(zip(input, gt_output, pytorch_output)):
            one_frame_shape = in_one.shape
            input_data = {
                'input': in_one.reshape([1, *one_frame_shape])
            }
            coreml_out = coreml_model.predict(input_data)['linear_87'].reshape(one_frame_shape)
            gt_out = gt_out.numpy()
            pytorch_out = pytorch_out.numpy()

            mpjpe1 = mpjpe(coreml_out, pytorch_out)
            mpjpe2 = mpjpe(pytorch_out, gt_out)
            mpjpe3 = mpjpe(coreml_out, gt_out)

            mpjpe_compare = np.concatenate((mpjpe_compare, mpjpe1))
            mpjpe_pytorch = np.concatenate((mpjpe_pytorch, mpjpe2))
            mpjpe_coreml = np.concatenate((mpjpe_coreml, mpjpe3))

        # Calculate the average errors
        avg_mpjpe_compare = np.mean(mpjpe_compare)
        avg_mpjpe_pytorch = np.mean(mpjpe_pytorch)
        avg_mpjpe_coreml = np.mean(mpjpe_coreml)

        # Print the results
        print(f"Average MPJPE between CoreML and PyTorch outputs: {avg_mpjpe_compare:.8f}")
        print(f"Average MPJPE between PyTorch and Ground Truth outputs: {avg_mpjpe_pytorch:.8f}")
        print(f"Average MPJPE between CoreML and Ground Truth outputs: {avg_mpjpe_coreml:.8f}")

