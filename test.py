import numpy as np

import torchsparse
import torch
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize, sparse_collate_tensors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mixed", action="store_true")
args = parser.parse_args()
mixed = args.mixed

COLOR_MAP = np.array(['#f59664', '#f5e664', '#963c1e', '#b41e50', '#ff0000',
                      '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff', '#ff96ff',
                      '#4b004b', '#4b00af', '#00c8ff', '#3278ff', '#00af00',
                      '#003c87', '#50f096', '#96f0ff', '#0000ff', '#ffffff'])

LABEL_MAP = np.array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 0, 1, 19,
                      19, 19, 2, 19, 19, 3, 19, 4, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 5, 6, 7, 19, 19, 19, 19, 19, 19,
                      19, 8, 19, 19, 19, 9, 19, 19, 19, 10, 11, 12, 13,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 14, 15, 16, 19, 19, 19, 19, 19,
                      19, 19, 17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
                      19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19])

# load sample lidar & label
lidar = np.fromfile('assets/000000.bin', dtype=np.float32)
label = np.fromfile('assets/000000.label', dtype=np.int32)
lidar = lidar.reshape(-1, 4)
label = LABEL_MAP[label & 0xFFFF]

# filter ignored points
lidar = lidar[label != 19]
label = label[label != 19]

# get rounded coordinates
coords = np.round(lidar[:, :3] / 0.05)
coords -= coords.min(0, keepdims=1)
feats = lidar

# sparse quantization: filter out duplicate points
indices, inverse = sparse_quantize(coords,
                                   feats,
                                   return_index=True,
                                   return_invs=True)
coords = coords[indices]
feats = feats[indices]

# construct the sparse tensor
inputs = SparseTensor(feats, coords)
inputs = sparse_collate_tensors([inputs]).cuda()

from model_zoo import spvnas_specialized

# load the model from model zoo
model = spvnas_specialized('SemanticKITTI_val_SPVNAS@65GMACs').cuda()
model.eval()

# run the inference
with torch.cuda.amp.autocast(enabled=mixed):
    print('inference=============================================')
    outputs = model(inputs)
    outputs = outputs.argmax(1).cpu().numpy()

# map the prediction back to original point clouds
outputs = outputs[inverse]

train_acc = np.sum(label == outputs)/len(label)
print(train_acc)