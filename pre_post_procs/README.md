# Preprocessing & Post Processing
Since our model learns the distribution of extracted features (i.e. spectrograms, cepstrograms, ...) while our dataset is consist of audio (.wav), we must first do some preprocessing (feature extraction). After the model generates the output, reversed feature-extration (post processing) is executed to get hearable results. 

## Preprocessing
We slice the audio into 3.5-second clips, and extract the timbre features clip by clip. Please read **Play_as_You_Like/pre_post_procs/audio2spectra.py** for detailed settings. The input/output paths are hard-coded, so you may want to make some changes before you run.
```
python audio2spectra.py
```
You may find some .jpg files as side products, which are visualized timbre features. They would be helping if you want to examine the quality of extracted features.

## Post Processing
Every generated piece of combined features is reconstructed to be a 3-second clip (precisely 256256 samples with sampling rate=22050). Every reconstructed clip is then concatenated, and finally outputed as one generated audio. Make sure the naming is in good order. This process needs to know the parameters of preprocessing (in .yaml format) and provided the origninal audio as phase information (so in this part, you need to know from which song the spectrograms are). Please carefully read the hard-coded input/output paths and you may still want to change them before you run.
```
python spectra2audio.py
```
After patiently waiting, don't forget to adjust the volume when you listen to the generated audio.