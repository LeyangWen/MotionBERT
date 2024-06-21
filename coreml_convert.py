import coremltools as ct
import torch
import pickle
import numpy as np
import argparse

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-e', '--evaluate', default='checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-o', '--out_path', type=str, help='eval pose output path', default=r'experiment/VEHS-7M_6D')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    # parser.add_argument('--test_set_keyword', default='test', type=str, help='eval set name, either test or validate, only for VEHS')
    # parser.add_argument('--wandb_project', default='MotionBert_train', type=str, help='wandb project name')
    # parser.add_argument('--wandb_name', default='VEHS_ft_train', type=str, help='wandb run name')
    # parser.add_argument('--note', default='', type=str, help='wandb notes')
    opts = parser.parse_args()
    return opts


def load_checkpoint(filename, model, map_location='cpu'):
    # Load the checkpoint
    checkpoint = torch.load(filename, map_location=map_location)
    # Remove 'module.' prefix if present (for DataParallel compatibility)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}
    # Load the state dict into your model
    model.load_state_dict(new_state_dict, strict=True)


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    model_backbone = load_backbone(args)
    input_output_file = r'edge/input_output.pkl'
    # Load the input/output shapes
    with open(input_output_file, 'rb') as f:
        input_output = pickle.load(f)
    input_tensor = input_output['input'].to(dtype=torch.float32)[0:1]
    gt_output_tensor = input_output['gt']
    input_ct = ct.TensorType(name='input_ct', shape=input_tensor.shape)


    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)
    load_checkpoint(chk_filename, model_backbone)
    print("Tracing model")
    model_backbone.eval()
    model_backbone.forward = model_backbone.forward_dummy
    traced_model = torch.jit.trace(model_backbone, input_tensor)

    # Load the PyTorch model
    # torch_model = torch.jit.load(torch_model_file, map_location=torch.device('cpu'))
    traced_model.eval()
    # todo: combine trace and script to get the model.forward

    print("Converting model")
    # Convert the model
    mlmodel = ct.convert(traced_model,
                         inputs=[input_ct],
                         source="pytorch",
                         convert_to="mlprogram",  # todo: look more here
                         )

    print("Saving model")
    # Save the converted model
    mlmodel.save('MyModel.mlmodel')
