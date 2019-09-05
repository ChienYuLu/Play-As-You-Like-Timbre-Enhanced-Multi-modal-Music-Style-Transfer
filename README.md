# Play as You Like: Timbre-Enhanced Multi-Modal Music Style Transfer

### Paper
[Chien-Yu Lu*](), [Min-Xin Xue*](), [Chia-Che Chang](http://chang810249.github.io), [Che-Rung Lee](http://www.cs.nthu.edu.tw/~cherung/), [Li Su](https://www.iis.sinica.edu.tw/pages/lisu/index_en.html), "[Play as You Like: Timbre-Enhanced Multi-Modal Music Style Transfer](https://arxiv.org/abs/1811.12214)", AAAI 2019

This is authors' pytorch implementation of the paper.

### How to Use

#### Dependency

```
python >= 3.6
pytorch >= 0.4.1
librosa >= 0.6.0
pyyaml
tensorboard
tensorboardX
```

#### Preprocessing and Post processing

[Code and tutorial.](pre_post_procs)

#### Training

1. Prepare your own dataset.
2. Setup the yaml file, see `configs/example.yaml` for more details.
3. Start training.
```
python train.py --config configs/example.yaml
```

#### Testing

You can run `test.py` after finishing training process. The following line will do an a to b style translation.
```
python test.py --config configs/example.yaml --input dataset/pia2vio_example/ --checkpoint outputs/example/checkpoints/gen.pt --a2b 1
```

### Results
![](https://i.imgur.com/HwBPOkF.png)

The left two columns are the input (original) and output (transferred) features of a piano to guitar transfer while the right two columns are features of a guitar to piano transfer. From top to bottom the features are: mel-spectrogram, MFCC, spectral difference, and spectral envelope.

Audio samples for a bilateral transfer of [piano to guitar](https://www.dropbox.com/s/ys9tipulzlpib5j/MUNIT-ALL-P2G-06.mp3?dl=0) and [guitar to piano](https://www.dropbox.com/s/rp395l9xritfvcp/MUNIT-ALL-G2P-04.mp3?dl=0).



Here's the [link](https://www.dropbox.com/sh/un0ws0aradjbxeq/AADR670aPJUCtemHJ-qt4aAja?dl=0) to all audio samples.

### Citation
```
@article{Lu_Xue_Chang_Lee_Su_2019,
  title={Play as You Like: Timbre-Enhanced Multi-Modal Music Style Transfer},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Lu, Chien-Yu and Xue, Min-Xin and Chang, Chia-Che and Lee, Che-Rung and Su, Li},
  year={2019},
  month={Jul.}
}
```

### Reference
> 1. [MUNIT](https://github.com/NVlabs/MUNIT)
> 2. [SPSI_Python](https://github.com/lonce/SPSI_Python)
