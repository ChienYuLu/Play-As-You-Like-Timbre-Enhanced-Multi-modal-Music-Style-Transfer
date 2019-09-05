import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
import yaml
import scipy
import librosa
import math
import time
import scipy.fftpack as fft
from scipy import signal
from scipy.io import wavfile

def mkdir(directory):
    if not os.path.exists(directory):
        print('making dir:{0}'.format(directory))
        os.makedirs(directory)
    else:
        print('already exist: {0}'.format(directory))
def read_via_scipy(file_name):
    sr, y = wavfile.read(file_name)
    y = np.divide(y, np.max(np.abs(y)), dtype='float32') # making the floating numbers
    if len(y.shape) != 1:
        y = np.mean(y, axis=1, dtype='float32')
    if sr != 22050: # librosa: to_mono then resample
        y = librosa.resample(y, sr, 22050)
        sr = 22050
    y_norm = np.divide(y, np.max(np.abs(y)), dtype='float32')
    return y_norm, sr

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def sendMsg2Console(msg):
    print('v'*10 + '\n' + msg + '\n' + '^'*10)

def ReLU(x):
    return np.maximum(x, 0.0)

def spsi(msgram, fftsize, hop_length) :
    """ https://github.com/lonce/SPSI_Python
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """
    
    numBins, numFrames  = msgram.shape
    y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
            m_mag=msgram[:, i] 
            for j in range(1,numBins-1) : 
                if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                    alpha=m_mag[j-1];
                    beta=m_mag[j];
                    gamma=m_mag[j+1];
                    denom=alpha-2*beta+gamma;
                    
                    if(denom!=0) :
                        p=0.5*(alpha-gamma)/denom;
                    else :
                        p=0;
                        
                    phaseRate=2*np.pi*(j+p)/fftsize;    #adjusted phase rate
                    m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                    peakPhase=m_phase[j];
                    
                    # If actual peak is to the right of the bin freq
                    if (p>0) :
                        # First bin to right has pi shift
                        bin=j+1;
                        m_phase[bin]=peakPhase+np.pi;
                        
                        # Bins to left have shift of pi
                        bin=j-1;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                            m_phase[bin]=peakPhase+np.pi;
                            bin=bin-1;
                        
                        #Bins to the right (beyond the first) have 0 shift
                        bin=j+2;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase;
                            bin=bin+1;
                            
                    #if actual peak is to the left of the bin frequency
                    if(p<0) :
                        # First bin to left has pi shift
                        bin=j-1;
                        m_phase[bin]=peakPhase+np.pi;

                        # and bins to the right of me - here I am stuck in the middle with you
                        bin=j+1;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase+np.pi;
                            bin=bin+1;
                        
                        # and further to the left have zero shift
                        bin=j-2;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                            m_phase[bin]=peakPhase;
                            bin=bin-1;
                            
                #end ops for peaks
            #end loop over fft bins with

            magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
            magphase[0]=0; magphase[numBins-1] = 0 #remove dc and nyquist
            m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
            
            #overlap and add
            m_recon=np.real(np.fft.ifft(m_recon))*m_win
            y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    return y_out


def spsi_eff(magD, y_out, fftsize, hop_length) :
    p = np.angle(librosa.stft(y_out, fftsize, hop_length, center=False))
    for i in range(50):
        S = magD * np.exp(1j*p)
        x = librosa.istft(S, hop_length, win_length=fftsize, center=True) # Griffin Lim, assumes hann window; librosa only does one iteration?
        p = np.angle(librosa.stft(x, fftsize, hop_length, center=True))
    return x

def magnitude2waveform(magD, config, phase, length=None):
    st = time.time()
    if phase is None:
        # reconstruct with phase estimation
        print('doing phase reconstruction...')
        y_out = spsi(magD, fftsize=config['fft_size'], hop_length=config['hop_length'])
        audio = spsi_eff(magD, y_out, fftsize=config['fft_size'], hop_length=config['hop_length'])        
    else: # phase is the D_phase of "D_mag,D_phase=librosa.magphase(stft_matrix)"
        # we should handle magD.shape != phaseD.shape
        # because magD may be padded zeros at the last spectrogram
        print('use original phase...')
        phaseD = np.ones_like(magD, dtype='complex64')
        min_hei = np.minimum(magD.shape[0], phase.shape[0])
        min_wid = np.minimum(magD.shape[1], phase.shape[1])
        if phase.dtype=='complex64':
            # only to adjust the shape
            phaseD[ :min_hei, :min_wid ] = phase[ :min_hei, :min_wid ]
        else: # if you don't get the phase information from librosa.magphase
		    # assuming using the imaginary part of stft-spectrogram (but not getting good results)
            phaseD[ :min_hei, :min_wid ] += phase[ :min_hei, :min_wid ]*1j
        stft_matrix = magD * phaseD # element-wise multiply
        audio = librosa.istft(stft_matrix, hop_length=config['hop_length'], length=length)
    ed = time.time()
    print('it costs {} seconds'.format(ed-st))
    return audio

def resolve_inputType_to_power(x, config):
    # resolve the process on [stft_matrix] or on [Mel-power spectrum]
    ret = x**(1.0/config['exp_b'])
    return ret

def spectrum2magnitude(spec, config):
    # in pre-processing, we may do something to a Mel-power spectrum
    # so first is to get Mel-power spectrums
    mel_pwr = resolve_inputType_to_power(spec, config)
    
    # and then use nnls to reconstruct a linear-frequency spectrum
    print('doing nnls...')
    melfb = librosa.filters.mel(22050, config['fft_size'], n_mels=config['n_mels']) # shape=(n_mels, 1 + n_fft/2)
    num_frames = mel_pwr.shape[1]
    pwr = np.zeros((1+config['fft_size']//2, num_frames))
    for i in range(num_frames):
        # reverse column by column
        ret = nnls(melfb, mel_pwr[:,i])
        pwr[:,i] = ret[0]
    # finally, get estimated magnitude
    mag = pwr**(0.5)
    return mag

### codes above are for audio-reconstruction from spectrograms
### codes below are for spectrogram-making
###
def power_to_outputType(pwr, config):
    ### the outputType here is inputType of training model
    return pwr**config['exp_b']

def chk_NaN(x):
    flag = False
    if np.isinf(x).any():
        print('inf error')
        flag = True
    if np.isnan(x).any():
        print('nan error')
        flag = True
    if flag==True:
        print('*'*10)
    

def get_spectrogram(data, config, win_len=None):
    ### get spectrogram according to the configuration and window_length
    ### we first calculate the power2-spectrum,
    ### and then get the Mel-spectrogram via the Mel-Filter banks
    stft_matrix = librosa.stft(data, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=win_len)
    mag_D = np.abs(stft_matrix)
    pwr = mag_D**2
    
    mel_basis = librosa.filters.mel(sr=config['sr'], n_fft=config['n_fft'], n_mels=config['n_mels'])
    mel_pwr = np.dot(mel_basis, pwr)
    chk_NaN(mel_pwr)
    # last, apply the gamma-power to approxiate Steven power law.
    return power_to_outputType(mel_pwr, config)
        
def get_cepstrogram(spec, config, win_len=None):
    ### return the cepstrogram according to the spectrogram
    ### spec.shape = [256,302]                 vvv
    mel_ceps_coef = scipy.fftpack.dct(spec, axis=0, type=config['dct_type'], norm=config['norm'])
    mel_ceps_coef_relu = np.maximum(mel_ceps_coef, 0.0)
    chk_NaN(mel_ceps_coef_relu)
    return mel_ceps_coef_relu

def get_diff_spectrogram(spec, mode=None):
    # only to diff by time
    d_spec = np.zeros_like(spec)
    hei, wid = d_spec.shape
    for i in range(1, wid-1):
        if mode=='all': # nxt - pre
            d_spec[:,i] = spec[:,i+1]-spec[:,i-1]
        elif mode=='decay': # ReLU(-all)
            d_spec[:,i] = ReLU(spec[:,i-1]-spec[:,i+1])
        elif mode=='attack': # ReLU(all)
            d_spec[:,i] = ReLU(spec[:,i+1]-spec[:,i-1])
    d_spec[:,0] = d_spec[:,1]
    d_spec[:,-1] = d_spec[:,-2]
    return d_spec

def get_spectral_envelope(mel_spec, config):
    MFCC = scipy.fftpack.dct(mel_spec, axis=0, type=config['dct_type'], norm=config['norm'])
    hei, wid = MFCC.shape
    MFCC[15:,:] = 0.0
    ret = scipy.fftpack.idct(MFCC, axis=0, type=config['dct_type'], norm=config['norm'])
    return ReLU(ret)

def plot_figure(img_name, spec_list, ceps_list, d_spec_list, spec_enve_list, config):
    ### to plot row = num_representation, col = num_resolution
    if (not spec_list):
        # the list is empty
        sendMsg2Console('Error !!! There are no spectra to plot')
        return
    num_presentation = 4
    if (not ceps_list):
        num_presentation -= 1
    if (not d_spec_list):
        num_presentation -= 1
    if (not spec_enve_list):
        num_presentation -= 1
    
    num_resolution = len(spec_list)
    hei, wid = spec_list[0].shape # 256, 302
    
    plt.figure(1, figsize=(7*(wid/302), 7/302*256*num_presentation)) # the resolution of y-axis goes up with hei
    plt.clf()
    for i in range(num_resolution):
        # plot spectrogram
        plt.subplot(num_presentation, num_resolution, i+1)
        librosa.display.specshow(spec_list[i], y_axis='mel', x_axis='time', hop_length=config['hop_length'])
        plt.colorbar()
        plt.title('channel '+ str(i) + ': win_len = ' + str(config['n_fft']//(2**i)))
        
        row_anchor = 1 # use row_anchor to plot ceps:Y/N, d_spec:Y/N
        if ceps_list:
            # plot cepstrogram                                  vvv i-th row           vvv i-th col
            plt.subplot(num_presentation, num_resolution, (row_anchor*num_resolution)+(i+1))
            librosa.display.specshow(ceps_list[i], x_axis='time', hop_length=config['hop_length'])
            plt.colorbar()
            plt.title('corresponding cepstrogram')
            row_anchor += 1
        if d_spec_list:
            # plot cepstrogram
            plt.subplot(num_presentation, num_resolution, (row_anchor*num_resolution)+(i+1))
            librosa.display.specshow(d_spec_list[i], y_axis='mel', x_axis='time', hop_length=config['hop_length'])
            plt.colorbar()
            plt.title('corresponding diff_spectrogram')
            #plt.colorbar(format='%+2.0f')
            row_anchor += 1
        if spec_enve_list:
            # plot spectral envelope
            plt.subplot(num_presentation, num_resolution, (row_anchor*num_resolution)+(i+1))
            librosa.display.specshow(spec_enve_list[i], y_axis='mel', x_axis='time', hop_length=config['hop_length'])
            plt.colorbar()
            plt.title('spectral envelope')
            row_anchor += 1
        
    plt.savefig(img_name, dpi='figure', bbox_inches='tight')
    plt.clf()
