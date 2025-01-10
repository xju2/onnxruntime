#!/usr/bin/env python

import onnx
import onnxruntime as ort
import psutil
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(1000, 1000)
        self.fc2 = torch.nn.Linear(1000, 1000)
        
    def forward(self, x, i: int = 50):
        for i in range(i):
            x = self.fc1(x)
            x = self.fc2(x)
            
        return x
    
model = Model()
model.eval()

input_data = torch.randn(1, 1000)

# convert to jit script model
with torch.jit.optimized_execution(True):
    script = torch.jit.script(model)

torch.jit.freeze(script)

torch.onnx.export(
    script, (input_data,), 'test.onnx', export_params=True, opset_version=11,
    do_constant_folding=False, input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

process = psutil.Process()
last = process.memory_info().rss
print(last/10**6)

# Load the ONNX model
onnx_model = onnx.load('test.onnx')
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession('test.onnx')

print(process.memory_info().rss/10**6, (process.memory_info().rss - last)/10**6)  # in bytes
