"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import scipy
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.optimize import nnls

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_loss
# slerp
# get_slerp_interp
# get_model_list            : get model list for resume
# get_scheduler
# weights_init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    
    train_loader_a, dataset_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
                                          new_size_a, height, width, num_workers, True, conf)
    test_loader_a, _ = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'), batch_size, False,
                                         new_size_a, height, width, num_workers, True, conf)
    train_loader_b, dataset_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
                                          new_size_b, height, width, num_workers, True, conf)
    test_loader_b, _ = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'), batch_size, False,
                                         new_size_b, height, width, num_workers, True, conf)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b, dataset_a, dataset_b


def generate_random_sample(dataset, batch_size):
    while True:
        random_indexes = np.random.choice(dataset.__len__(), size=batch_size, replace=False)
        batch = [dataset[i] for i in random_indexes]
        yield torch.stack(batch, 0)


def get_test_data_loaders(conf, input_path, a2b):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if a2b == 1:
        test_loader, _ = get_data_loader_folder(os.path.join(input_path, 'testA'), batch_size, False,
                                             new_size_a, new_size_a, new_size_a, num_workers, True, conf)
    else:    
        test_loader, _ = get_data_loader_folder(os.path.join(input_path, 'testB'), batch_size, False,
                                             new_size_b, new_size_b, new_size_b, num_workers, True, conf)
    return test_loader


def __RandomCropNumpy(x, output_size):
    h, w, c = x.shape
    #print('hello {0}'.format(x.shape)) # 256 302 3
    th, tw = output_size
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    res = x[i:i+th, j:j+tw, :]
    return res


def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True, conf=None):
    transform_list = []
    transform_list = [transforms.ToTensor()] + transform_list
    transform_list = [transforms.Lambda(lambda x: __RandomCropNumpy(x, (height, width)))] + transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader, dataset


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def write2npy(outputs, display_image_num, file_name, config):
    # file_name = .../a2b_test_
    names = ['x', 'recon', 'trans_fix', 'trans_rand']
    
    for name_idx, img in enumerate(outputs):
        # img is ndarray of 4, n_channel, 256, 256
        for idx in range(4):
            if idx>=display_image_num:
                continue
            npy_name = file_name + str(idx).zfill(2) + '_' + names[name_idx] + '.npy'
            npy_var = img[idx] # n_channel, 256, 256
            np.save(npy_name, npy_var)


def write2spec(outputs, display_image_num, file_name, config, feature_type):
    # file_name = ".../a2b_test_spec_"
    # only handles a channel
    # so expected to have a list of [4,256,256] * 4
    fig_name = file_name + '.png'
    #print('writing to ', fig_name)
    row = len(outputs)
    col = 4 # it should be display_image_num, but train_current will crash so it's a fixed '4'
    
    if row==3: # UNIT
        #names = ['x', 'recon', 'trans']
        plt.figure(num=1, figsize=(16,10)) # UNIT
    elif row==4: # MUNIT
        #names = ['x', 'recon', 'trans_fix', 'trans_rand']
        plt.figure(num=1, figsize=(16,13.33)) # MUNIT
    
    plt.clf()
    
    idx_row = -1
    for img in outputs:
        idx_row += 1
        idx_col = 0
        for idx in range(4):
            idx_col += 1
            x = img[idx] # img[idx] is a [256, 256]
            # now it's time to tranform it to dB and plot
            # but now default is using a pseudo-dB unit
            # because no power^0.3, it's too large to learn
            plt.subplot(row, col, idx_row*col+idx_col)
            if feature_type=='ceps':
                # the y-axis for cepstrums is quefrency
                librosa.display.specshow(x, x_axis='time', hop_length=config['hop_length'])
            else: # it's spec or diff_spec or spec_enve
                if ('is_mel' in config) and (config['is_mel']==True):
                    librosa.display.specshow(x, x_axis='time', y_axis='mel', hop_length=config['hop_length'])
                else:
                    librosa.display.specshow(x, x_axis='time', y_axis='linear', hop_length=config['hop_length'])
    plt.tight_layout(pad=-0.4, w_pad=0.0, h_pad=0.0)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()
    

def write2figures(image_outputs, a2b_name, b2a_name, display_image_num, config, feature_type):
    n = len(image_outputs)
    write2spec(image_outputs[0:n//2], display_image_num, a2b_name, config, feature_type)
    write2spec(image_outputs[n//2:n], display_image_num, b2a_name, config, feature_type)
    

###        write_2audio
###         /       \
###        /         \
###     write2npy    write2figures
###                      \
###                    write2spec
def write_2audio(image_outputs, display_image_num, image_directory, postfix, config):
    ### image_outputs is a list of 4+4 tensors
    print('write to audio...')
    img_list = [ image.cpu().numpy() for image in image_outputs]
    ### img_list is a list of 4+4 ndarray ([a2b + b2a] X [x,recon,fix_style,rand_style]), 
    ### each [display_num],[num_channels],256,256
    ###          [4, ch=3, 256, 256]
    
    num_ch = (img_list[0].shape)[1]
    n = len(img_list)
    display_num = display_image_num
    if postfix == 'train_current':
        display_num = 1
    
    a2b_file_name = '{}/gen_a2b_{}_'.format(image_directory, postfix) # ".../a2b_test_" or ".../a2b_train_current_"
    b2a_file_name = '{}/gen_b2a_{}_'.format(image_directory, postfix) # ".../b2a_test_" or ".../b2a_train_current_"
        
    ### save npy, throw the concerns at npy2wav
    write2npy(img_list[0:n//2], display_num, a2b_file_name, config)
    write2npy(img_list[n//2:n], display_num, b2a_file_name, config)
    
    channel_anchor = 0
    # plot spec
    spectrums = [ img[:,channel_anchor,:,:] for img in img_list] # now spectrums is [8][4,x,256,256]=[8][4,256,256]
    a2b_spec_name = a2b_file_name + 'spec_'
    b2a_spec_name = b2a_file_name + 'spec_'
    write2figures(spectrums, a2b_spec_name, b2a_spec_name, display_num, config, 'spec')
    channel_anchor += 1
    if ('use_ceps' in config) and (config['use_ceps']):
        cepstrums = [ img[:,channel_anchor,:,:] for img in img_list] # now spectrums is [8][4,256,256]
        a2b_ceps_name = a2b_file_name + 'ceps_'
        b2a_ceps_name = b2a_file_name + 'ceps_'
        write2figures(cepstrums, a2b_ceps_name, b2a_ceps_name, display_num, config, 'ceps')
        channel_anchor += 1
    if ('use_diff_spec' in config) and (config['use_diff_spec']):
        diff_spectrums = [ img[:,channel_anchor,:,:] for img in img_list] # now spectrums is [8][4,256,256]
        a2b_dspec_name = a2b_file_name + 'diff_spec_'
        b2a_dspec_name = b2a_file_name + 'diff_spec_'
        write2figures(diff_spectrums, a2b_dspec_name, b2a_dspec_name, display_num, config, 'diff_spec')
        channel_anchor += 1
    if ('use_spec_enve' in config) and (config['use_spec_enve']):
        diff_spectrums = [ img[:,channel_anchor,:,:] for img in img_list] # now spectrums is [8][4,256,256]
        a2b_dspec_name = a2b_file_name + 'spec_enve'
        b2a_dspec_name = b2a_file_name + 'spec_enve'
        write2figures(diff_spectrums, a2b_dspec_name, b2a_dspec_name, display_num, config, 'spec_enve')
        channel_anchor += 1
    
    while channel_anchor < num_ch:
        # equal to config['is_multi']
        print('lazy to multi... ', num_ch-channel_anchor)
        channel_anchor += 1


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun