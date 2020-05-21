#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# PROGRAMMER: Janani
# Imports here

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import json
import args


def main():
    in_arg = args.get_input_args_predict()
    print("Arguments: ", in_arg)

    def load_the_checkpoint(path):
        load_dict_checkpt = torch.load(path)
        print("Model arch:", load_dict_checkpt['arch'])
        if load_dict_checkpt['arch'] == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif load_dict_checkpt['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
        model.classifier = load_dict_checkpt['classifier']
        model.epochs = load_dict_checkpt['epoch']
        model.class_to_idx = load_dict_checkpt['class_to_idx']
        model.load_state_dict(load_dict_checkpt['state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        return model

    model = load_the_checkpoint(in_arg.checkpoint)
    if in_arg.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'

    def process_image(image):
        im_process = Image.open(image)
        if im_process.size[0] < im_process.size[1]:
            new_width = 256
            im_process.thumbnail((new_width, im_process.size[1]), Image.ANTIALIAS)
        else:
            new_height = 256
            im_process.thumbnail((im_process.size[0], new_height), Image.ANTIALIAS)

        left_pt = (im_process.size[0] - 224) / 2
        right_pt = left_pt + 224
        top_pt = (im_process.size[1] - 224) / 2
        bottom_pt = top_pt + 224
        im_process = im_process.crop((left_pt, top_pt, right_pt, bottom_pt))
        np_image = np.array(im_process)
        np_image = np_image / 255
        means = [0.485, 0.456, 0.406]
        standard_deviations = [0.229, 0.224, 0.225]
        np_image = np_image - means
        np_image = np_image / standard_deviations
        np_image = np.transpose(np_image, (2, 0, 1))
        return np_image

    def predict(image_path, model, top_k=5):

        image_input = process_image(image_path)
        image_input = torch.from_numpy(image_input).float().to(device)
        image_input = image_input.unsqueeze(0)
        model.to(device)
        model.eval()
        output = model(image_input)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k)
        top_p = top_p.detach().cpu().numpy().tolist()[0]
        top_class = top_class.detach().cpu().numpy().tolist()[0]
        idx_to_class = {}
        for key, value in model.class_to_idx.items():
            idx_to_class[value] = key
        top_c = []
        for clas in top_class:
            top_c.append(idx_to_class[clas])

        return top_p, top_c

    probs, classes = predict(in_arg.image_path, model, top_k = in_arg.top_k)


    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    class_list = []
    for clas in classes:
        class_list.append(cat_to_name[clas])

    for i, j in zip(probs, class_list):
        print("Flower :{} has a probability: {}".format(j, i))
        
#     print("Probabilities: \n", probs)
#     print("Flower Names: \n", class_list)


if __name__ == "__main__":
    main()