"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import get_test_data_loaders

import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, default='test', help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of pre-trained model")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--task_name',type=str, default='', help="task name, default will be config file name")


opts = parser.parse_args()

print('*'*20)
print('config=', opts.config)
print('input=', opts.input)
print('output_folder=', opts.output_folder)
print('checkpoint=', opts.checkpoint)
print('a2b=', opts.a2b)
print('num_style=', opts.num_style)
print('seed=', opts.seed)
print('*'*20)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
style_dim = config['gen']['style_dim']
trainer = MUNIT_Trainer(config)

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()
# encode function
encode = trainer.gen_a.encode if opts.a2b == 1 else trainer.gen_b.encode
style_encode = trainer.gen_b.encode if opts.a2b == 1 else trainer.gen_a.encode
# decode function
decode = trainer.gen_b.decode if opts.a2b == 1 else trainer.gen_a.decode # decode function

st = time.time()

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b == 1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

test_loader = get_test_data_loaders(config, opts.input, opts.a2b)
style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda(), volatile=True) if opts.style != '' else None
iterations = opts.checkpoint[-11:-3]
if opts.task_name == '':
    task_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_folder, task_name+'_'+iterations)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

a2b_directory = os.path.join(output_directory, 'a2b')
b2a_directory = os.path.join(output_directory, 'b2a')
if not os.path.exists(a2b_directory):
    os.makedirs(a2b_directory)
if not os.path.exists(b2a_directory):
    os.makedirs(b2a_directory)

# Start testing
style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
for j in range(opts.num_style):
    style_npy_var = style_rand[j].cpu().numpy()
    if opts.a2b == 1:
        style_name = a2b_directory + '/style_code_' + str(j).zfill(2)
    else:
        style_name = b2a_directory + '/style_code_' + str(j).zfill(2)
    np.save(style_name, style_npy_var)
for it, (data) in enumerate(test_loader):
    images = data
    images = Variable(images.cuda())
    if opts.style != '':
        with torch.no_grad():
            _, style = style_encode(style_image)
    else:
        style = style_rand

    with torch.no_grad():
        content, _ = encode(images)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            x =  outputs[0].cpu().numpy()
            if opts.a2b == 1:
                npy_name = a2b_directory + '/piece'+str(it).zfill(4) + '_style_' + str(j).zfill(2)
            else:
                npy_name = b2a_directory + '/piece'+str(it).zfill(4) + '_style_' + str(j).zfill(2)
            np.save(npy_name, x)

ed = time.time()
print("elapsed {0} seconds".format(ed-st))