#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# PROGRAMMER: Janani
# Imports here
import argparse

def get_input_args_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, default='flowers', help='specify path to the folder of flower images (mandatory argument)')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='specify path to the folder to save checkpoint')
    parser.add_argument('--arch', type=str, default='densenet121', help='specify the architecture to use (densenet121 or vgg16)')
    parser.add_argument('--hidden', type=int, default=500, help='specify the hidden unit value for hidden layer')
    parser.add_argument('--epoch', type=int, default=5, help='specify the no. of epochs to run')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='specify the learning rate')
    parser.add_argument('--gpu', type=str, default='gpu', help='specify gpu or cpu to use')

    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, default='flowers/test/5/image_05166.jpg', help='specify the input image path (mandatory argument)')

    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help='specify path to the checkpoint (mandatory argument)')
    parser.add_argument('--top_k', type=int, default=5, help='specify the no. of top classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='specify the json file map between classes and flower names')
    parser.add_argument('--gpu', type=str, default='gpu', help='specify gpu or cpu to use')

    return parser.parse_args()

