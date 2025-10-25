import os
from fnmatch import filter as fnf
from time import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import timm
import torch
from monai.transforms import Compose
from monai.transforms import NormalizeIntensity
from monai.transforms import Resize
from monai.transforms import ResizeWithPadOrCrop
from torch import nn
from tqdm import tqdm


class SAMViTClassifier(nn.Module):
    def __init__(self, in_chans=1, num_classes=7, class_values=3, model_name='vit_base_patch16_224'):
        super(SAMViTClassifier, self).__init__()

        # Load pre-trained SAM-ViT encoder from timm
        self.encoder = timm.create_model(model_name, in_chans=in_chans, pretrained=False)

        encoder_dim = self.encoder.num_features

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, num_classes * class_values)
        )

        self.num_classes = num_classes
        self.class_values = class_values

    def forward(self, x):
        features = self.encoder(x)

        logits = self.head(features)

        # Reshape to [B, num_classes, class_values]
        logits = logits.view(-1, self.num_classes, self.class_values)

        return logits


def get_lst(path):
    contents = []
    for f in fnf(os.listdir(path), "*.nii.gz"):
        contents.append({'image_path': os.path.join(path, f),
                         "pid": f.replace('.nii.gz', '')})
    return contents


def read_series(d):
    if os.path.isdir(d):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(d))
        image = reader.Execute()
    else:
        image = sitk.ReadImage(d)
    return image


if __name__ == '__main__':
    data_dir = '/input'
    out_dir = '/output'
    ckpt_path = '/weight.pth.tar'

    log_itv = 5

    data_list = get_lst(data_dir)

    label_names = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]

    # model
    model = SAMViTClassifier(in_chans=1, num_classes=7, class_values=3)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda()

    records = []

    t1 = time()
    bt1 = time()
    for idx, item in enumerate(data_list):
        imitk = read_series(item['image_path'])
        img = np.expand_dims(sitk.GetArrayFromImage(imitk), axis=0)
        reintense = NormalizeIntensity()
        img = reintense(img).squeeze(0)

        slice_record = []

        slice_fn = Compose([
            Resize(spatial_size=224, size_mode='longest'),
            ResizeWithPadOrCrop(spatial_size=(224, 224)),
        ])

        for sid, slice in enumerate(img):
            slice = slice.unsqueeze(0)
            slice = slice_fn(slice)
            slice = slice.unsqueeze(0).cuda()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    out = model(slice).squeeze(0)
            out = out.detach().cpu()
            slice_record.append(out)
        slice_record = torch.stack(slice_record)
        slice_record = slice_record.softmax(-1).max(dim=0)[0].argmax(-1).tolist()

        basename = os.path.basename(item['image_path']).replace('.nii.gz', '')

        out_item = {'filename': basename,
                    }
        for name, o in zip(label_names, slice_record):
            out_item[name] = o

        if 'label' in item.keys():
            out_item['label'] = list(item['label'].values())

        records.append(out_item)

        if idx % max(1, (len(data_list) // log_itv)) == 0:
            print(f"{100 * idx / len(data_list):02.02f}% {time() - bt1} s/image")
        bt1 = time()

    records = pd.DataFrame(records)
    records.to_csv(os.path.join(out_dir, "LISA_LF_QC_predictions.csv"), index=False)
