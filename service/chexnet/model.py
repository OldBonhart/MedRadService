map_location = 'cpu'
import os
import json
import torch
import numpy
import torchvision.transforms as transforms
import torch
from PIL import Image

import base64
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import torch.nn as nn

import numpy as np
from .dataset import ChestXrayDataSet

from io import BytesIO
import matplotlib as mpl
from matplotlib import cm



all_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
              'Infiltration', 'Mass',  'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax']


## Uploading model
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 15)
f = 'service/chexnet/resnet18.pt'
resnet18.load_state_dict(torch.load(f, map_location='cpu'))
resnet18 = resnet18.cpu()

# Save features from model
class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0, :, :, ].reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

def preprocessing(img, sigmaX=10):

    inp_img = np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                 std=[0.229, 0.224, 0.225])
        ])

    img_tensor = preprocess(img).unsqueeze(0).cpu()
    return img_tensor, inp_img


def make_predict(image):
    # Load predtrained Model
    model = resnet18
    ### Last layer's features for heatmap
    final_layer = model._modules.get('layer4')
    activated_features = SaveFeatures(final_layer)
    ###
    img_tensor, in_img = preprocessing(image)
    blind_prediction = model(img_tensor)
    probabilities = F.softmax(blind_prediction, dim=1).data.squeeze()
    label = np.argmax(probabilities.cpu().detach().numpy())
    #print(probabilities, '---', label)
    activated_features.remove()

    ## weights
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

    ### Heat Map of ROI overlay
    overlay = getCAM(activated_features.features, weight_softmax, label)
    #overlay = resize(overlay, (300, 300))
    cm_hot = mpl.cm.get_cmap('rainbow') # color map jet
    overlay = cm_hot(overlay)
    overlay = np.uint8(overlay * 255)
    overlay = Image.fromarray(overlay).convert("RGB")
    overlay = overlay.resize((312, 312), Image.BILINEAR)
    in_img = Image.fromarray(in_img)#.convert("RGB")
    # New image by interpolating between two images,
    # using a constant alpha.
    heatmap = Image.blend(in_img, overlay, 0.5)
    stream = BytesIO()
    heatmap.save(stream, format='PNG') # load img to byte
    stream.flush()
    stream.seek(0)
    heatmap = base64.b64encode(stream.getvalue()) # load the bytes in the context as base64

    probabilities = probabilities.cpu().detach().numpy()
    proba = []
    for i, class_name in enumerate(all_labels):
        proba.append([str(class_name), str(np.round(probabilities[i]*100, 2)) + '%'])
        
    diagnosis = proba[label]
    #print(diagnosis)
    #print(label)
    #print(proba)
    return heatmap, proba, diagnosis


