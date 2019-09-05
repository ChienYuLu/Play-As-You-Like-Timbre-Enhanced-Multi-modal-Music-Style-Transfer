"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2audio, generate_random_sample
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
trainer = MUNIT_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b, dataset_a, dataset_b = get_all_data_loaders(config)

if config['dis']['gan_type'] == 'ralsgan':
    random_sample_a = generate_random_sample(dataset_a, config['batch_size'])
    random_sample_b = generate_random_sample(dataset_b, config['batch_size'])

train_display_data_a = torch.stack([train_loader_a.dataset[i].clone() for i in range(display_size)])
train_display_data_b = torch.stack([train_loader_b.dataset[i].clone() for i in range(display_size)])
test_display_data_a = torch.stack([test_loader_a.dataset[i].clone() for i in range(display_size)])
test_display_data_b = torch.stack([test_loader_b.dataset[i].clone() for i in range(display_size)])

train_display_images_a = train_display_data_a.cuda()
train_display_images_b = train_display_data_b.cuda()
test_display_images_a = test_display_data_a.cuda()
test_display_images_b = test_display_data_b.cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (data_a, data_b) in enumerate(zip(train_loader_a, train_loader_b)): # to iterate along both lists
        trainer.update_learning_rate()
        images_a = data_a
        images_b = data_b
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        # Main training code
        trainer.dis_update(images_a, images_b, config)        
        if config['dis']['gan_type'] == 'ralsgan':
            images_rand_a = random_sample_a.__next__()
            images_rand_b = random_sample_b.__next__()
            images_rand_a, images_rand_b = Variable(images_rand_a.cuda()), Variable(images_rand_b.cuda())
            trainer.gen_update(images_a, images_b, config, images_rand_a, images_rand_b)
        else:
            trainer.gen_update(images_a, images_b, config)
        torch.cuda.synchronize()
        trainer.update_iter()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Model: %s, Iteration: %08d/%08d" % (model_name, iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Training logs
        if (iterations + 1) % config['image_save_iter'] == 0:
            iter_directory = os.path.join(output_directory+'/images', 'iter_'+str(iterations + 1).zfill(8))
            if not os.path.exists(iter_directory):
                print("Creating directory: {}".format(iter_directory))
                os.makedirs(iter_directory)
            # Test set logs
            image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
            write_2audio(image_outputs, display_size, iter_directory, 'test_%08d' % (iterations + 1), config)
            # Train set logs
            image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2audio(image_outputs, display_size, iter_directory, 'train_%08d' % (iterations + 1), config)

        if (iterations + 1) % config['image_display_iter'] == 0:
            image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2audio(image_outputs, display_size, image_directory, 'train_current', config)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

