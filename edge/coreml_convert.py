import coremltools as ct
import torch
import pickle

input_output_file = r'input_output.pkl'
torch_model_file = r'model.pt'

# Load the input/output shapes
with open(input_output_file, 'rb') as f:
    input_output = pickle.load(f)

# Load the PyTorch model
torch_model = torch.load(torch_model_file)

# Convert the model
mlmodel = ct.convert(torch_model,
                     inputs=[ct.ImageType(shape=(1, 1, H, W), color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
    ],
    minimum_deployment_target=ct.target.macOS13,
)

# Save the converted model
mlmodel.save('MyModel.mlmodel')
