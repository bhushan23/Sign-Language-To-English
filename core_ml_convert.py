# import sys
# from onnx import onnx_pb
# from onnx_coreml import convert

# model_in = sys.argv[1]
# model_out = sys.argv[2]

# model_file = open(model_in, 'rb')
# model_proto = onnx_pb.ModelProto()
# model_proto.ParseFromString(model_file.read())
# coreml_model = convert(model_proto, image_input_names=['inputImage'], image_output_names=[])
# coreml_model.save(model_out)

import onnx
from onnx_coreml import convert
model_in = './checkpoints/onnx_model_name.onnx'

onnx_model = onnx.load(model_in)
# model_file = open('checkpoints/onnx_model_name.onnx', 'rb')
# onnx_model = onnx_pb.ModelProto()
# onnx_model.ParseFromString(model_file.read())

print(onnx.helper.printable_graph(onnx_model.graph))

coreml_model = convert(model=onnx_model) #,
                           # add_custom_layers=True)
coreml_model.save('coreml_model.mlmodel')
