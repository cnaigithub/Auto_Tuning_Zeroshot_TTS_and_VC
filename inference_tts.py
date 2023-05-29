import os
import torch
from scipy.io import wavfile
from tqdm import tqdm
import argparse
import numpy as np

import utils
import commons
from models import SynthesizerTrn
from text_total import SYMBOLS, _symbols_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", "-ckpt", type=str)
parser.add_argument("--config", "-cfg", type=str, default="configs/english.json")
parser.add_argument("--filelist", "-f", type=str)
parser.add_argument("--target", "-t", type=str)
parser.add_argument("--output", "-o", default="results/")
parser.add_argument("--language", "-l", type=str, default="english")

args = parser.parse_args()

hps = utils.get_hparams_from_file(args.config)
if hps.model.append_lang_emb:
    lang2int = {x: i for i, x in enumerate(hps.data.language_list)}

# Load model and checkpoint
net_g = SynthesizerTrn(
    hps,
    len(SYMBOLS) + getattr(hps.data, "add_blank", False),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).to(device)
net_g.eval()

model_state_dict, optimizer_state_dict, global_step = utils.load_checkpoint(
    args.checkpoint_path
)
net_g.load_state_dict(model_state_dict, strict=False)

# Inference given text
with open(args.filelist, "r") as in_file:
    lines = [x.strip() for x in in_file.readlines()]

os.makedirs(args.output, exist_ok=True)

ref_wav = args.target
ref_spec = utils.get_spec_inference(ref_wav, hps, device)
ref_spec_lengths = torch.LongTensor([ref_spec.shape[2]]).to(device)

lang = (
    torch.LongTensor([lang2int[args.language]]).to(device)
    if hps.model.append_lang_emb
    else None
)

for idx, line in enumerate(tqdm(lines)):
    text = line.strip()

    cleaned_text = _symbols_to_sequence(text)
    if hps.data.add_blank:
        cleaned_text = commons.intersperse(cleaned_text, len(SYMBOLS))
    cleaned_text = torch.LongTensor(cleaned_text).to(device).unsqueeze(0)

    with torch.no_grad():
        o, attn, y_mask, (z, z_p, m_p, logs_p) = net_g.infer(
            cleaned_text,
            torch.LongTensor([cleaned_text.shape[1]]).to(device),
            ref_spec=ref_spec,
            ref_spec_lengths=ref_spec_lengths,
            lang=lang,
        )

    audio = o[0][0].data.cpu().float().numpy() * 32768.0
    audio = audio.astype(np.int16)

    wavfile.write(
        os.path.join(args.output, f"{idx}_voice_synthesized.wav"),
        hps.data.sampling_rate,
        audio,
    )
