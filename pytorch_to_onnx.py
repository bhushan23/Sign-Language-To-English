import torch
import torch.onnx
from models import *

# A model class instance (class not shown)
model = M2()

# Load the weights from a file (.pth usually)
state_dict = torch.load('./checkpoints/model95.pkl')

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 3, 64, 64)

torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")