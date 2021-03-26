# smartula-analysis

Repository for seraching optimal bee-sound represenation with use of Autoencoders:

* Vanilla Autoencoder (ae)
* Convolutional Autoencoder (conv_ae)
* Variational Autoencoder (vae)
* Convolutional Autoencoder (conv_vae)
* Contrastive Autoencoder (cvae)
* Contrastive Convolutional Autoencoder (conv_cvae)

and sound representation features:

* Periodogram (periodogram)
* spectrogram (spectrogram)
* MelSpectrogram (melspectrogram)
* MFCC (mfcc)
* Bioacustic indicies (indicies)
  * Acustic Complexity Index (aci - configured by config.json)
  * Acoustic Diversity Index (adi - configured by config.json)
  * Acoustic Evenness Index (aei - configured by config.json)
  * Bioacustic Index (bi - configured by config.json)

## Training defined model 

### non contrastive 
train.py scripts trains choosen architecture with use config from json file. E.g. command will invoke training Variational Autoencoder on periodogram data.

```bash
python train.py vae periodogram ..\\measurements\\smartulav2 --config_file=example_config.json
```

### constrastive
user should specify target and background data through options
```bash
python train.py vae periodogram ..\\measurements\\smartulav2 --config_file=example_config.json --target smrpiclient6 --background smrpiclient3 smrpiclient7
```
also there is option to use discriminator in contrastive training by using ```--discriminator``` or ```--no-discriminator``` option


