import os
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_revgrad import RevGrad


import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text_total import SYMBOLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_step = 0


class LagLambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.lag_lambda = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.RevGrad = RevGrad()

    def forward(self):
        return self.RevGrad(self.lag_lambda)


if __name__ == "__main__":
    hps = utils.get_hparams()
    writer = SummaryWriter(log_dir=os.path.join(hps.output_dir, "runs"))
    train_dataset = TextAudioLoader(hps, hps.data.training_files)
    val_dataset = TextAudioLoader(hps, hps.data.validation_files)
    collate_fn = TextAudioCollate(hps)
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        batch_size=hps.train.batch_size,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=8,
        shuffle=True,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    net_g = (
        SynthesizerTrn(
            hps,
            len(SYMBOLS) + getattr(hps.data, "add_blank", False),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )
        .to(device)
        .train()
    )

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device).train()
    optim_g = torch.optim.AdamW(
        [parameter for parameter in net_g.parameters() if parameter.requires_grad],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if hps.train.use_damped_lagrangian:
        lagrangian_lambda = LagLambda().train().to(device)
        optim_lambda = torch.optim.AdamW(
            lagrangian_lambda.parameters(),
            hps.train.lambda_lr,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    latest_G_checkpoint = utils.latest_checkpoint_path(hps.output_dir, "G_*.pth")
    latest_D_checkpoint = utils.latest_checkpoint_path(hps.output_dir, "D_*.pth")
    if latest_G_checkpoint is None or latest_D_checkpoint is None:
        global_step = 0
    else:
        G_state_dict, G_optimizer_state_dict, global_step = utils.load_checkpoint(
            latest_G_checkpoint
        )
        D_state_dict, D_optimizer_state_dict, global_step = utils.load_checkpoint(
            latest_D_checkpoint
        )
        net_g.load_state_dict(G_state_dict, strict=False)
        optim_g.load_state_dict(G_optimizer_state_dict)
        net_d.load_state_dict(D_state_dict)
        optim_d.load_state_dict(D_optimizer_state_dict)
    if hps.train.use_damped_lagrangian:
        latest_lambda = utils.latest_checkpoint_path(hps.output_dir, "lambda_*.pth")
        if latest_lambda is not None:
            lambda_ckpt = torch.load(latest_lambda)
            lagrangian_lambda.load_state_dict(lambda_ckpt["lambda"])
            optim_lambda.load_state_dict(lambda_ckpt["lambda_optim"])

    for epoch in range(hps.train.n_epochs):
        for (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            lang,
        ) in train_loader:
            (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                lang,
            ) = (
                text_padded.to(device),
                text_lengths.to(device),
                spec_padded.to(device),
                spec_lengths.to(device),
                wav_padded.to(device),
                wav_lengths.to(device),
                lang.to(device),
            )

            try:
                (
                    y_hat,
                    l_length,
                    attn,
                    ids_slice,
                    x_mask,
                    y_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(text_padded, text_lengths, spec_padded, spec_lengths, lang)
            except KeyboardInterrupt:
                print("Keyboard interrupt")
                exit(1)
            except Exception as e:
                print(e)
                continue

            mel = spec_to_mel_torch(
                spec_padded,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = commons.slice_segments(
                wav_padded, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            optim_d.zero_grad()
            loss_disc_all.backward()
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            optim_d.step()

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)

            loss_dur = torch.sum(l_length.float())
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

            if hps.train.use_damped_lagrangian:
                damp = (
                    hps.train.damping * (hps.train.epsilon_mel_loss - loss_mel).detach()
                )
                loss_gen_all = (
                    loss_gen
                    + loss_fm
                    + loss_dur
                    + loss_kl * hps.train.c_kl
                    - hps.train.c_mel
                    * (lagrangian_lambda() - damp)
                    * (hps.train.epsilon_mel_loss - loss_mel)
                )
            else:
                loss_gen_all = (
                    loss_gen
                    + loss_fm
                    + loss_mel * hps.train.c_mel
                    + loss_dur
                    + loss_kl * hps.train.c_kl
                )

            optim_g.zero_grad()
            if hps.train.use_damped_lagrangian:
                optim_lambda.zero_grad()
            loss_gen_all.backward()
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)

            optim_g.step()
            if hps.train.use_damped_lagrangian:
                optim_lambda.step()
                if lagrangian_lambda.lag_lambda < 0:
                    lagrangian_lambda.lag_lambda = (
                        lagrangian_lambda.lag_lambda - lagrangian_lambda.lag_lambda
                    )
            if global_step % hps.train.log_interval == 0:
                scalar_dict = {
                    "train.loss_gen_all": loss_gen_all.item(),
                    "train.loss_kl": loss_kl.item(),
                    "train.loss_fm": loss_fm.item(),
                    "train.loss_gen": loss_gen.item(),
                    "train.disc_loss": loss_disc.item(),
                    "train.loss_mel": loss_mel.item(),
                    "train.loss_dur": loss_dur.item(),
                }
                if hps.train.use_damped_lagrangian:
                    scalar_dict.update(
                        {"train.lambda": lagrangian_lambda.lag_lambda.item()}
                    )
                utils.summarize(writer, global_step, scalar_dict)
                print(f"Step: {global_step} Loss: {loss_gen_all.item():.5f}")

            if global_step % hps.train.eval_interval == 0:
                net_g.eval()
                net_d.eval()
                (
                    mel_list,
                    kl_list,
                    fm_list,
                    gen_list,
                    dur_list,
                    gen_all_list,
                    disc_list,
                ) = ([], [], [], [], [], [], [])

                for (
                    text_padded,
                    text_lengths,
                    spec_padded,
                    spec_lengths,
                    wav_padded,
                    wav_lengths,
                    lang,
                ) in val_loader:
                    (
                        text_padded,
                        text_lengths,
                        spec_padded,
                        spec_lengths,
                        wav_padded,
                        wav_lengths,
                        lang,
                    ) = (
                        text_padded.to(device),
                        text_lengths.to(device),
                        spec_padded.to(device),
                        spec_lengths.to(device),
                        wav_padded.to(device),
                        wav_lengths.to(device),
                        lang.to(device),
                    )
                    with torch.no_grad():
                        try:
                            (
                                y_hat,
                                l_length,
                                attn,
                                ids_slice,
                                x_mask,
                                y_mask,
                                (z, z_p, m_p, logs_p, m_q, logs_q),
                            ) = net_g(
                                text_padded,
                                text_lengths,
                                spec_padded,
                                spec_lengths,
                                lang,
                            )
                        except KeyboardInterrupt:
                            print("Keyboard interrupt")
                            exit(1)
                        except Exception as e:
                            print(e)
                            continue
                        mel = spec_to_mel_torch(
                            spec_padded,
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                        y_mel = commons.slice_segments(
                            mel,
                            ids_slice,
                            hps.train.segment_size // hps.data.hop_length,
                        )
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                        y = commons.slice_segments(
                            wav_padded,
                            ids_slice * hps.data.hop_length,
                            hps.train.segment_size,
                        )  # slice

                        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                            y_d_hat_r, y_d_hat_g
                        )
                        loss_disc_all = loss_disc

                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                        loss_dur = torch.sum(l_length.float())
                        loss_mel = F.l1_loss(y_mel, y_hat_mel)
                        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
                        loss_fm = feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_gen_all = (
                            loss_gen
                            + loss_fm
                            + loss_mel * hps.train.c_mel
                            + loss_dur
                            + loss_kl * hps.train.c_kl
                        )
                        mel_list.append(loss_mel.item())
                        kl_list.append(loss_kl.item())
                        fm_list.append(loss_fm.item())
                        dur_list.append(loss_dur.item())
                        gen_list.append(loss_gen.item())
                        gen_all_list.append(loss_gen_all.item())
                        disc_list.append(loss_disc.item())
                scalar_dict = {
                    "val.loss_gen_all": utils.mean(gen_all_list),
                    "val.loss_kl": utils.mean(kl_list),
                    "val.loss_fm": utils.mean(fm_list),
                    "val.loss_gen": utils.mean(gen_list),
                    "val.disc_loss": utils.mean(disc_list),
                    "val.loss_mel": utils.mean(mel_list),
                    "val.loss_dur": utils.mean(dur_list),
                }

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }

                utils.summarize(
                    writer, global_step, scalars=scalar_dict, images=image_dict
                )

                net_g.train()
                net_d.train()
            if global_step % hps.train.checkpoint_interval == 0:
                torch.save(
                    {
                        "model_state_dict": net_g.state_dict(),
                        "optimizer_state_dict": optim_g.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(hps.output_dir, f"G_{global_step}.pth"),
                )
                torch.save(
                    {
                        "model_state_dict": net_d.state_dict(),
                        "optimizer_state_dict": optim_d.state_dict(),
                        "global_step": global_step,
                    },
                    os.path.join(hps.output_dir, f"D_{global_step}.pth"),
                )
                if hps.train.use_damped_lagrangian:
                    torch.save(
                        {
                            "lambda": lagrangian_lambda.state_dict(),
                            "lambda_optim": optim_lambda.state_dict(),
                        },
                        os.path.join(hps.output_dir, f"lambda_{global_step}.pth"),
                    )
            global_step += 1
