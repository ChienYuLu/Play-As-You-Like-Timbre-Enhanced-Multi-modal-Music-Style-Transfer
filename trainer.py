"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np
import scipy.fftpack


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']

        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
         
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), eps=1e-8, weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.iter = 0

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def intrinsic_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def volumeloss_criterion(self, input, target):
        idx_select =  torch.tensor([0]).cuda()
        input, target = input.index_select(1, idx_select), target.index_select(1, idx_select)
        input, target = torch.mean(input, 3), torch.mean(target, 3)
        input, target = torch.mean(input, 2), torch.mean(target, 2)
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        with torch.no_grad():
            self.eval()
            s_a = Variable(self.s_a)
            s_b = Variable(self.s_b)
            c_a, s_a_fake = self.gen_a.encode(x_a)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
            self.train()
            return x_ab, x_ba

    def update_iter(self):
        self.iter += 1

    def gen_update(self, x_a, x_b, hyperparameters, x_a_rand=None, x_b_rand=None):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba, x_a_rand)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab, x_b_rand)
        # ceps loss
        self.loss_gen_ceps_a = self.calc_cepstrum_loss(x_ba) if hyperparameters['ceps_w'] > 0 else 0
        self.loss_gen_ceps_b = self.calc_cepstrum_loss(x_ab) if hyperparameters['ceps_w'] > 0 else 0
        # flux loss
        self.loss_gen_flux_a2b = self.calc_spectral_flux_loss(x_ab) if hyperparameters['flux_w'] > 0 else 0
        self.loss_gen_flux_b2a = self.calc_spectral_flux_loss(x_ba) if hyperparameters['flux_w'] > 0 else 0
        # enve loss
        self.loss_gen_enve_a2b = self.calc_spectral_enve15_loss(x_ab) if hyperparameters['enve_w'] > 0 else 0
        self.loss_gen_enve_b2a = self.calc_spectral_enve15_loss(x_ba) if hyperparameters['enve_w'] > 0 else 0
        # volume loss
        self.loss_gen_vol_a = self.volumeloss_criterion(x_a, x_ab) if hyperparameters['vol_w'] > 0 else 0
        self.loss_gen_vol_b = self.volumeloss_criterion(x_b, x_ba) if hyperparameters['vol_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['ceps_w'] * self.loss_gen_ceps_a + \
                              hyperparameters['ceps_w'] * self.loss_gen_ceps_b + \
                              hyperparameters['flux_w'] * self.loss_gen_flux_a2b + \
                              hyperparameters['flux_w'] * self.loss_gen_flux_b2a + \
                              hyperparameters['enve_w'] * self.loss_gen_enve_a2b + \
                              hyperparameters['enve_w'] * self.loss_gen_enve_b2a + \
                              hyperparameters['vol_w'] * self.loss_gen_vol_a + \
                              hyperparameters['vol_w'] * self.loss_gen_vol_b
        self.loss_gen_total.backward()        
        if hyperparameters['clip_grad'] == 'value':
            torch.nn.utils.clip_grad_value_(list(self.gen_a.parameters()) + list(self.gen_b.parameters()), 1)
        elif hyperparameters['clip_grad'] == 'norm':
            torch.nn.utils.clip_grad_norm_(list(self.gen_a.parameters()) + list(self.gen_b.parameters()), 0.5)
        self.gen_opt.step()

    def calc_cepstrum_loss(self, x_fake):
        idx_select_spec = torch.tensor([0]).cuda()
        idx_select_ceps = torch.tensor([1]).cuda()

        fake_spec = x_fake.index_select(1, idx_select_spec).detach().cpu().numpy()
        ceps = scipy.fftpack.dct(fake_spec, axis=2, type=2, norm='ortho')
        ceps = np.maximum(ceps, 0)        
        return self.intrinsic_criterion(x_fake.index_select(1, idx_select_ceps), torch.from_numpy(ceps).cuda())

    def calc_spectral_flux_loss(self, x_fake):
        idx_select_spec = torch.tensor([0]).cuda()
        idx_select_flux = torch.tensor([2]).cuda()

        fake_spec = x_fake.index_select(1, idx_select_spec).detach().cpu().numpy()
        spec_flux = np.zeros_like(fake_spec)
        hei, wid = 256, 256
        for i in range(1, wid-1):
            spec_flux[:,:,:,i] = np.maximum(fake_spec[:,:,:,i+1]-fake_spec[:,:,:,i-1], 0.0)
        spec_flux[:,:,:,0] = spec_flux[:,:,:,1]
        spec_flux[:,:,:,-1] = spec_flux[:,:,:,-2]
        return self.intrinsic_criterion(x_fake.index_select(1, idx_select_flux), torch.from_numpy(spec_flux).cuda())

    def calc_spectral_enve15_loss(self, x_fake):
        idx_select_spec = torch.tensor([0]).cuda()
        idx_select_enve = torch.tensor([3]).cuda()
        
        fake_spec = x_fake.index_select(1, idx_select_spec).detach().cpu().numpy()		
        MFCC = scipy.fftpack.dct(fake_spec, axis=2, type=2, norm='ortho')
        MFCC[:,:,15:,:] = 0.0
        spec_enve = scipy.fftpack.idct(MFCC, axis=2, type=2, norm='ortho')
        spec_enve = np.maximum(spec_enve, 0.0)
        return self.intrinsic_criterion(x_fake.index_select(1, idx_select_enve), torch.from_numpy(spec_enve).cuda())

    def sample(self, x_a, x_b):
        with torch.no_grad():
            self.eval()
            s_a1 = Variable(self.s_a)
            s_b1 = Variable(self.s_b)
            s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
            x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
            x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
            x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
            x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
            self.train()
            return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        #self.save_grad(list(self.dis_a.named_parameters()) + list(self.dis_b.named_parameters()))
        #torch.nn.utils.clip_grad_norm_(list(self.dis_a.parameters()) + list(self.dis_b.parameters()), 0.5)
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
