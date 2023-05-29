import os
import torch
from scipy.io import wavfile
import subprocess
from tqdm import tqdm
import argparse
import numpy as np

import utils
from models import SynthesizerTrn
from text_total import SYMBOLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", "-ckpt", type=str)
parser.add_argument("--config", "-cfg", type=str, default="configs/english.json")
parser.add_argument("--filelist", "-f", type=str)
parser.add_argument("--target", "-t", type=str)
parser.add_argument("--output", "-o", default="results/")
parser.add_argument("--ignore_layers", "-il", nargs="+", default=["enc_p", "dp."])

args = parser.parse_args()

hps = utils.get_hparams_from_file(args.config)
# Load model and checkpoint
net_g = SynthesizerTrn(
    hps,
    len(SYMBOLS) + getattr(hps.data, "add_blank", False),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).to(device)
net_g.eval()
net_g, global_step = utils.warm_start_model(
    args.checkpoint_path, net_g, args.ignore_layers
)

with open(args.filelist, "r") as in_file:
    lines = [x.strip() for x in in_file.readlines()]

os.makedirs(args.output, exist_ok=True)

target_wav = args.target
target_spec = utils.get_spec_inference(target_wav, hps, device)
target_spec_lengths = torch.LongTensor([target_spec.shape[2]]).to(device)

for idx, line in enumerate(tqdm(lines)):
    audio_path = line.strip()

    spec = utils.get_spec_inference(audio_path, hps, device)
    spec_lengths = torch.LongTensor([spec.shape[2]]).to(device)

    with torch.no_grad():
        o_hat, y_mask, (z, z_p, z_hat) = net_g.voice_conversion(
            source_spec=spec,
            source_spec_lengths=spec_lengths,
            target_spec=target_spec,
            target_spec_lengths=target_spec_lengths,
        )

    audio = o_hat[0][0].data.cpu().float().numpy() * 32768.0
    audio = audio.astype(np.int16)

    wavfile.write(
        os.path.join(args.output, f"{idx}_voice_synthesized.wav"),
        hps.data.sampling_rate,
        audio,
    )
