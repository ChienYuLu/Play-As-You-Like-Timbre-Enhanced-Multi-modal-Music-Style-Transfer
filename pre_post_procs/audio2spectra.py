import matplotlib
matplotlib.use('Agg') # librosa.display includes matplotlib
import librosa.display
import numpy as np
import glob

from utils import mkdir, read_via_scipy, get_spectrogram, get_cepstrogram, get_diff_spectrogram, get_spectral_envelope, plot_figure, ReLU

### settings
config = {
    # basic parameters
    'sr': 22050,
    'n_fft': 2048,
    'hop_length': 256,
    'input_type': 'exp', # power, dB with ref_dB, p_log, exp with exp_b. it's input of training data
    'is_mel': True,
    
    # for spectra
    'n_mels': 256,
    'exp_b': 0.3,
    'ref_dB': 1e-5,
    
    # for cepstrum
    'dct_type': 2,
    'norm': 'ortho',
    
    # for slicing and overlapping
    'audio_samples_frame_size': 77175, # 3.5sec * sr
    'audio_samples_hop_length': 77175,
    'output_hei': 256,
    'output_wid': 302, # num_output_frames = 1 + (77175/hop_length256)
    
    # to decide number of channels
    'use_phase': False, # only True without mel
    'is_multi': False, # if true, there would be three resolutions
    'use_ceps': True,
    'use_d_spec': False,
    'd_spec_type': 'attack', # mode: all, decay, or attack
    'use_spec_enve': False,
    
    'num_digit': 4
}
print(config)

###
def cal_num_channels(config):
    n_ch = 1
    n_res = 1
    win_lens = [config['n_fft']]
    if config['is_multi']:
        n_res = 3
        win_lens.append(config['n_fft']//2)
        win_lens.append(config['n_fft']//4)
    if config['use_ceps']:
        n_ch += 1
    if config['use_phase']:
        n_ch += 1
    if config['use_d_spec']:
        n_ch += 1
    if config['use_spec_enve']:
        n_ch += 1
    
    return n_ch*n_res, win_lens

def audio2npys(input_file, config):
    # read an audio file and then write a lot of numpy files
    song_name = input_file.split('/')[-1][:-4]
    print('!song_name = {}!'.format(song_name))
    
    y, sr = read_via_scipy(input_file)
    print("dtype={}, sampling rate={}, len_samples={}".format(y.dtype, sr, len(y)))
    num_ch, mul_win_len = cal_num_channels(config)
    print('num_ch = {}, mul_win_len={}'.format(num_ch, mul_win_len))
    
    Len = y.shape[0]
    cnt = 0
    st_idx = 0
    ed_idx = st_idx+config['audio_samples_frame_size']
    nxt_idx = st_idx+config['audio_samples_hop_length']
    
    while st_idx<Len:
        if ed_idx>Len:
            ed_idx = Len
        data = np.zeros(config['audio_samples_frame_size'], dtype='float32')
        data[:ed_idx-st_idx] = y[st_idx:ed_idx]
        
        out_var = np.zeros((num_ch, config['output_hei'], config['output_wid']), dtype='float32')
        
        list_spec = []
        list_ceps = []
        list_d_spec = []
        list_spec_enve = []
        channel_anchor = 0 # use this to save thourgh out_var[:,hei,wid]
        for idx, w_len in enumerate(mul_win_len):
            # config['is_multi'] is decided by "current for-loop"
            list_spec.append(get_spectrogram(data, config, w_len))
            out_var[channel_anchor] = list_spec[-1]
            channel_anchor += 1
            if config['use_ceps']:
                list_ceps.append(get_cepstrogram(list_spec[-1], config, w_len))
                out_var[channel_anchor] = list_ceps[-1]
                channel_anchor += 1
            if config['use_d_spec']:
                # mode: all, decay, or attack
                list_d_spec.append(get_diff_spectrogram(list_spec[-1], mode=config['d_spec_type']))
                out_var[channel_anchor] = list_d_spec[-1]
                channel_anchor += 1
            if config['use_spec_enve']:
                list_spec_enve.append(get_spectral_envelope(list_spec[-1], config))
                out_var[channel_anchor] = list_spec_enve[-1]
                channel_anchor += 1
        #print('channel_anchor = ', channel_anchor, num_ch)
        npy_name = specpath+song_name+'_'+str(cnt).zfill(config['num_digit'])+'.npy'
        #print('cnt ={}, max={}'.format(cnt, np.max(list_spec[-1])))
        
        np.save(npy_name, out_var)
        img_name = imgpath+song_name+'_'+str(cnt).zfill(config['num_digit'])+'.png'
        
        # plots: 1. spec 2. ceps (all in single file)
        plot_figure(img_name, list_spec, list_ceps, list_d_spec, list_spec_enve, config)
        
        cnt += 1
        st_idx = nxt_idx
        ed_idx = st_idx+config['audio_samples_frame_size']
        nxt_idx = st_idx+config['audio_samples_hop_length']


'''
locate the input directory & read the files
'''
inpath = './raw_audios/raw_audio_'
instrument = 'guitar'
prefix = '_c2h256w302' # naming follows the settings in config

specpath = prefix+'_'+instrument+'/npy'+'/'
imgpath = prefix+'_'+instrument+'/img'+'/'
mkdir(specpath)
mkdir(imgpath)

input_dir = inpath + instrument + '/'
print('from {0}'.format(input_dir))
ls = sorted(glob.glob(input_dir+'/*.wav'))

for file in ls:
    print('file = ', file)
    audio2npys(file, config)
