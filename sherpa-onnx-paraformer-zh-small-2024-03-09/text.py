import onnx
model = onnx.load("model.int8.onnx")
print(model.ir_version)