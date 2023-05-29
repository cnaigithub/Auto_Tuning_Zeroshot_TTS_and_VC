import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text_total import SYMBOLS, _symbols_to_sequence


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, hps, filelist_path):
        self.hps = hps

        if hps.model.append_lang_emb:
            language_list = hps.data.language_list
            self.language_to_int = {x: i for i, x in enumerate(language_list)}

        self.audiopaths_and_text = load_filepaths_and_text(filelist_path)

        self.max_wav_value = hps.data.max_wav_value
        self.sampling_rate = hps.data.sampling_rate
        self.filter_length = hps.data.filter_length
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self.sampling_rate = hps.data.sampling_rate

        self.add_blank = hps.data.add_blank
        self.min_text_len = getattr(hps.data, "min_text_len", 1)
        self.max_text_len = getattr(hps.data, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """

        audiopaths_and_text_new = []
        for line in self.audiopaths_and_text:
            text = line[1]
            if (
                self.min_text_len <= len(text)
                and len(text) <= self.max_text_len
                and "(en)" not in text
            ):
                audiopaths_and_text_new.append(line)
        self.audiopaths_and_text = audiopaths_and_text_new

    def get_audio_text(self, audiopath_and_text):
        if self.hps.model.append_lang_emb:
            lang = audiopath_and_text[2]
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)

        return (
            text,
            spec,
            wav,
            torch.LongTensor(
                [self.language_to_int[lang] if self.hps.model.append_lang_emb else -1]
            ),
        )

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio_norm.unsqueeze(0)

        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)

        return spec, audio_norm

    def get_text(self, text):
        text_norm = []

        text_norm = _symbols_to_sequence(text)

        if self.add_blank:
            text_norm = commons.intersperse(text_norm, len(SYMBOLS))
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, hps, return_ids=False):
        self.return_ids = return_ids
        self.hps = hps

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        lang = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            lang[i] = row[3]

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            lang,
        )
