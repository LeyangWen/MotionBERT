import coremltools as ct
import torch
import torch.onnx
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


if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    # args.backbone = 'DSTformer_coreml'  # modified backbone
    model_backbone = load_backbone(args)
    input_output_file = r'edge/input_output.pkl'
    # Load the input/output shapes
    with open(input_output_file, 'rb') as f:
        input_output = pickle.load(f)
    input_tensor = input_output['input'].to(dtype=torch.float32)[0:1]
    gt_output_tensor = input_output['gt']


    chk_filename = opts.evaluate
    print('Loading checkpoint', chk_filename)
    load_checkpoint(chk_filename, model_backbone)
    model_backbone.eval()
    print("Tracing model")
    ##### tracing
    traced_model = torch.jit.trace(model_backbone, input_tensor)
    # torch_model = torch.jit.load(torch_model_file, map_location=torch.device('cpu'))

    ##### scripting
    # scripted_model = torch.jit.script(model_backbone)

    # traced_model == scripted_model

    ##### ONNX
    # onnx_model = torch.onnx.export(traced_model, input_tensor, 'model.onnx', verbose=False)
    # # model = ct.convert(onnx_model, inputs=[input_ct], source="pytorch")  # need old converter



    # print("Converting model")
    # # Convert the model
    input_ct = ct.TensorType(name='input_ct', shape=input_tensor.shape)
    input_ct_2 = ct.TensorType(name='input_ct_2', shape=input_tensor.shape)
    mlmodel = ct.convert(traced_model,
                         inputs=[input_ct],
                         source="pytorch",
                         convert_to="mlprogram"
                         )
    # mlmodel = ct.convert(scripted_model,
    #                      inputs=[input_ct, input_ct_2],
    #                      source="pytorch",
    #                      convert_to="mlprogram"
    #                      )  # https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert

    print("Saving model")
    # # Save the converted model
    mlmodel.save('MyModel.mlpackage')

    coreml_model = ct.models.MLModel('MyModel.mlpackage')
    input_data = {
        'input_ct': input_tensor  # Replace with actual input data and name
    }
    output = coreml_model.predict(input_data)
