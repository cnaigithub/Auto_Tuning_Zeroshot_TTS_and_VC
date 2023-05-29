# Auto_Tuning_Zeroshot_TTS_and_VC
PyTorch implementation of our [paper](https://arxiv.org/abs/2305.16699) "Automatic Tuning of Loss Trade-offs without Hyper-parameter Search in End-to-End Zero-Shot Speech Synthesis", accepted at INTERSPEECH 2023.\
[Demo page](https://cnaigithub.github.io/SpeechDewarping/)


> **Abstract:** 
Recently, zero-shot TTS and VC methods have gained attention due to their practicality of being able to generate voices even unseen during training.
Among these methods, zero-shot modifications of the VITS model have shown superior performance, while having useful properties inherited from VITS.
However, the performance of VITS and VITS-based zero-shot models vary dramatically depending on how the losses are balanced.
This can be problematic, as it requires a burdensome procedure of tuning loss balance hyper-parameters to find the optimal balance.
In this work, we propose a novel framework that finds this optimum without search, by inducing the decoder of VITS-based models to its full reconstruction ability.
With our framework, we show superior performance compared to baselines in zero-shot TTS and VC, achieving state-of-the-art performance.
Furthermore, we show the robustness of our framework in various settings.
We provide an explanation for the results in the discussion.

<!-- <strong> The repository is currently under construction.</strong> -->
The code is based on the [VITS](https://github.com/jaywalnut310/vits) repository.

## Installation
We tested our code in Ubuntu 20.04, CUDA 11.7 and Python 3.7.11 enviroment with A6000 GPUs.
```
conda create -n auto python=3.7.11
conda activate auto
pip install -r requirements.txt
cd monotonic_align; mkdir monotonic_align; python setup.py build_ext --inplace
sudo apt-get install espeak-ng
pip install phonemizer
```

## Dataset
For details about the dataset, please refer to the paper.\
You may use custom datasets also.\
Using the VCTK dataset will work fine.


### Pre-processing
You need paired audio, text pairs.\
Preprocess all audios to a sampling rate of 16000Hz.


### Filelists
Follow the given filelist format for each line of the file.
- {Audio file path}|{Text}\
Once the filelist is made in this format, it has to be phonemized (i.e. converted to IPA phonemes).

### Phonemizing the filelist
```
python phonemize.py -i {original txt filelist path} -o {output txt filelist path} -l {language code}
```
For the language code option, refer to the [phonemizer repository](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md).

For monolingual training, each line of the filelist should have this format:\
{audiopath}|{phonemized text}\
(e.g PATH_TO_VCTK/p277/p277_203.wav|twˈɑːdəl ɪz ɐ kˈeɪs ɪn pˈɔɪnt.)

For multilingual training, each line of the filelist should have this format:\
{audiopath}|{phonemized text}|{language}\
(e.g PATH_TO_VCTK/p277/p277_203.wav|twˈɑːdəl ɪz ɐ kˈeɪs ɪn pˈɔɪnt.|english)\
(You can also this filelist format for monolingual training also, but the language will be ignored.)


## Training
Use one of the json config files under `./configs`.\
Write your filelist paths at data.training_files / data.validation_files.

Here are explanations for some options in the conifg file.
- train.use_damped_lagrangian: whether to use our proposed MDMM based optimization
- train.epsilon_mel_loss: the user chosen value for $\varepsilon$, the target value of reconstruction loss. As a result of our paper, 0.25 should work fine. If you want to obtain $\varepsilon^*$ yourself, use the [HiFi-GAN repository](https://github.com/jik876/hifi-gan). Note that you will have to change their mel spectraogram code to the one in this repository.
- data.training_files / validation_files: path to the phonemized filelist (each for training and validation)
- data.language_list: The list of all languages used in training. The ordering within this list should be same during training and inference. This list is ignored if model.append_lang_emb is false.
- model.append_lang_emb: Whether to enable multilingual training or not. If the dataset is monolingual, set to false. Otherwise set to true.


Then run:
```
python train.py -o {Output folder to save checkpoints and logs} -c {Path to config file}
```

## Inference
For inference, you need a test filelist formatted as following:
- For TTS inference: each line should contain phonemized text only.
- For VC inference: each line should contain the source audiopath only.


The target voice should be converted to have a sampling rate of 16khz. (Important)

Run \
```python inference_tts.py -ckpt {saved checkpoint path} -cfg {config file used for training} -f {test filelist} -t {target voice audiopath} -o {directory to store results} -l {language of the text in the filelist}```\
(the -l option does not matter when model.append_lang_emb is false.)\
or\
```python inference_vc.py -ckpt {saved checkpoint path} -cfg {config file used for training} -f {test filelist} -t {target voice audiopath} -o {directory to store results}```

## Pre-trained Checkpoints
We provide the following [checkpoints](https://drive.google.com/file/d/1GSJ-bJQCa1GN9N0XjUHSbQVZjq0toiWQ/view?usp=sharing):\
Discriminator, generator and the Lagrangian coefficient checkpoint, trained for 500k steps with the VCTK dataset, using $\varepsilon=0.25$. ('Zero-shot VITS with our framework' in Table1.)\
The generator can be used for inference, with the given config file (./configs/english.json).