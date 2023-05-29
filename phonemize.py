import phonemizer
import argparse
from tqdm import tqdm
import pykakasi


def hanja2hira(text, kks):
    result = kks.convert(text)
    conv = ""
    for item in result:
        conv += item["hira"]
    return conv


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-o", "--output", type=str)
parser.add_argument("-l", "--lang", type=str)
parser.add_argument("-n", "--njobs", type=int, default=6)
args = parser.parse_args()

global_phonemizer = phonemizer.backend.EspeakBackend(
    language=args.lang, preserve_punctuation=True, with_stress=True
)

with open(args.input, "r") as infile:
    lines = [x.strip() for x in infile.readlines()]

texts = [x.split("|")[1] for x in lines]

if args.lang == "ja":
    kks = pykakasi.kakasi()
    texts = [hanja2hira(text, kks) for text in tqdm(texts)]
phonemes = global_phonemizer.phonemize(texts, strip=True, njobs=args.njobs)

with open(args.output, "w") as outfile:
    for line, phoneme in zip(lines, phonemes):
        if "(en)" not in phoneme:
            audiopath = line.split("|")[0]
            outfile.write(f"{audiopath}|{phoneme}\n")
