import os
import math
import tqdm
import torch
import itertools
import traceback
from utils.validation import  validate
from model.generator import Generator
from model.multiscale import MultiScaleDiscriminator
from model.mpd import MPD
from .utils import get_commit_hash
from utils.stft_loss import MultiResolutionSTFTLoss
import numpy as np
from utils.stft import TacotronSTFT


def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print('Trainable Parameters: %.3fM' % parameters)

def train(args, pt_dir, chkpt_path, trainloader, valloader, writer, logger, hp, hp_str):
    model_g = Generator(hp.audio.n_mel_channels).cuda()
    model_d = MultiScaleDiscriminator(hp.model.num_D, hp.model.ndf, hp.model.n_layers,
                                      hp.model.downsampling_factor, hp.model.disc_out).cuda()
    model_d_mpd = MPD().cuda()

    optim_g = torch.optim.AdamW(model_g.parameters(),
                               lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.AdamW(itertools.chain(model_d.parameters(), model_d_mpd.parameters()),
                               lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
   

    stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                        hop_length=hp.audio.hop_length,
                        win_length=hp.audio.win_length,
                        n_mel_channels=hp.audio.n_mel_channels,
                        sampling_rate=hp.audio.sampling_rate,
                        mel_fmin=hp.audio.mel_fmin,
                        mel_fmax=hp.audio.mel_fmax)

    githash = get_commit_hash()

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        model_d_mpd.load_state_dict(checkpoint['model_d_mpd'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint. Will use new.")

        if githash != checkpoint['githash']:
            logger.warning("Code might be different: git hash is different.")
            logger.warning("%s -> %s" % (checkpoint['githash'], githash))

    else:
        logger.info("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.adam.lr_decay, last_epoch=init_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hp.train.adam.lr_decay, last_epoch=init_epoch)

    try:
        model_g.train()
        model_d.train()
        stft_loss = MultiResolutionSTFTLoss()
        criterion = torch.nn.MSELoss().cuda()
        l1loss = torch.nn.L1Loss()


        for epoch in itertools.count(init_epoch + 1):
            if epoch % hp.log.validation_interval == 0:
                with torch.no_grad():
                    validate(hp, model_g, model_d, model_d_mpd, valloader, stft_loss, l1loss, criterion, stft, writer,
                             step)

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            avg_g_loss = []
            avg_d_loss = []
            avg_adv_loss = []
            for (melG, audioG), (melD, audioD) in loader:
                melG = melG.cuda()  # torch.Size([16, 80, 64])
                audioG = audioG.cuda()  # torch.Size([16, 1, 16000])
                melD = melD.cuda()  # torch.Size([16, 80, 64])
                audioD = audioD.cuda()  # torch.Size([16, 1, 16000]

                # generator
                optim_g.zero_grad()
                fake_audio = model_g(melG)[:, :, :hp.audio.segment_length]  # torch.Size([16, 1, 12800])

                loss_g = 0.0



                sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audioG.size(2)].squeeze(1), audioG.squeeze(1))
                loss_g += sc_loss + mag_loss # STFT Loss

                adv_loss = 0.0
                loss_mel = 0.0
                if step > hp.train.discriminator_train_start_steps:

                    # for multi-scale discriminator
                    disc_real_scores, disc_real_feats = model_d(audioG)
                    disc_fake_scores, disc_fake_feats = model_d(fake_audio)

                    for score_fake in disc_fake_scores:
                        # adv_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                        adv_loss += criterion(score_fake, torch.ones_like(score_fake))
                    adv_loss = adv_loss / len(disc_fake_scores)  # len(disc_fake) = 3

                    # MPD Adverserial loss
                    mpd_real_scores, mpd_real_feats = model_d_mpd(audioG)
                    mpd_fake_scores, mpd_fake_feats = model_d_mpd(fake_audio)

                    for score_fake in mpd_fake_scores:
                        adv_mpd_loss = criterion(score_fake, torch.ones_like(score_fake))
                    adv_mpd_loss = adv_mpd_loss / len(mpd_fake_scores)
                    adv_loss = adv_loss + adv_mpd_loss # Adv Loss

                    # Mel Loss
                    mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
                    loss_mel += l1loss(melG[:, :, :mel_fake.size(2)], mel_fake.cuda()) # Mel L1 loss
                    loss_g += hp.model.lambda_mel * loss_mel

                    if hp.model.feat_loss:
                        for feats_fake, feats_real in zip(disc_fake_feats, disc_real_feats):
                            for feat_f, feat_r in zip(feats_fake, feats_real):
                                adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

                        for feats_fake, feats_real in zip(mpd_fake_feats, mpd_real_feats):
                            for feat_f, feat_r in zip(feats_fake, feats_real):
                                adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))




                    loss_g += hp.model.lambda_adv * adv_loss

                loss_g.backward()
                optim_g.step()

                # discriminator
                loss_d_avg = 0.0
                if step > hp.train.discriminator_train_start_steps:
                    fake_audio = model_g(melD)[:, :, :hp.audio.segment_length]
                    fake_audio = fake_audio.detach()
                    loss_d_sum = 0.0
                    for _ in range(hp.train.rep_discriminator):
                        optim_d.zero_grad()
                        disc_fake_scores, _ = model_d(fake_audio)
                        disc_real_scores, _ = model_d(audioD)
                        loss_d = 0.0

                        # MSD
                        for score_fake, score_real in zip(disc_fake_scores, disc_real_scores):
                            loss_d_real = criterion(score_real, torch.ones_like(score_real))
                            loss_d_fake = criterion(score_fake, torch.zeros_like(score_fake))
                        loss_d_real = loss_d_real / len(disc_real_scores)  # len(disc_real) = 3
                        loss_d_fake = loss_d_fake / len(disc_fake_scores)  # len(disc_fake) = 3
                        loss_d += loss_d_real + loss_d_fake # MSD loss
                        loss_d_sum += loss_d

                        # MPD Adverserial loss
                        mpd_fake_scores, _ = model_d_mpd(fake_audio)
                        mpd_real_scores, _ = model_d_mpd(audioD)
                        for score_fake, score_real in zip(mpd_fake_scores, mpd_real_scores):
                            loss_mpd_real = criterion(score_real, torch.ones_like(score_real))
                            loss_mpd_fake = criterion(score_fake, torch.zeros_like(score_fake))
                        loss_mpd = (loss_mpd_fake + loss_mpd_real)/len(mpd_real_scores) # MPD Loss
                        loss_d += loss_mpd

                        loss_d.backward()
                        optim_d.step()
                        loss_d_sum += loss_mpd


                    loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                    loss_d_avg = loss_d_avg.item()

                step += 1
                # logging
                loss_g = loss_g.item()
                avg_g_loss.append(loss_g)
                avg_d_loss.append(loss_d_avg)
                avg_adv_loss.append(adv_loss)

                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    logger.error("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % hp.log.summary_interval == 0:
                    writer.log_training(loss_g, loss_d_avg, adv_loss, loss_mel, step)
                    loader.set_description(
                        "Avg : g %.04f d %.04f ad %.04f| step %d" % (sum(avg_g_loss) / len(avg_g_loss),
                                                                     sum(avg_d_loss) / len(avg_d_loss),
                                                                     sum(avg_adv_loss) / len(avg_adv_loss),
                                                                     step))
            if epoch % hp.log.save_interval == 0:
                save_path = os.path.join(pt_dir, '%s_%s_%04d.pt'
                                         % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'model_d_mpd': model_d_mpd.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

            scheduler_g.step()
            scheduler_d.step()

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
