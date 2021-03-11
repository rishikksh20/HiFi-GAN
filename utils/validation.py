import tqdm
import torch


def validate(hp, generator, discriminator, model_d_mpd, valloader, stft_loss, l1loss, sub_stft_loss, criterion,  pqmf,
             stft, writer, step):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for mel, audio, c in loader:
        mel = mel.cuda()
        audio = audio.cuda()    # B, 1, T torch.Size([1, 1, 212893])
        c = c.type(torch.cuda.FloatTensor)
        c = c.transpose(1, 2)
        #print(c.shape)


        # generator

        fake_audio = generator(mel, c) # B, 1, T' torch.Size([1, 1, 212992])
        if hp.model.out_channels > 1:
            y_mb_ = fake_audio
            fake_audio = pqmf.synthesis(y_mb_)[:, :, :audio.size(2)]
        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)])  # B, 1, T torch.Size([1, 1, 212893])
        disc_real = discriminator(audio)

        adv_loss = 0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        loss_mpd_real = 0.0
        loss_mpd_fake = 0.0
        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss

        if hp.model.use_subband_stft_loss:
            loss_g *= 0.5  # for balancing with subband stft loss
            y_mb = pqmf.analysis(audio)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            sub_sc_loss, sub_mag_loss = sub_stft_loss(y_mb_[:, :y_mb.size(-1)], y_mb)
            loss_g += 0.5 * (sub_sc_loss + sub_mag_loss)

        # Mel Loss

        mel_fake = stft.mel_spectrogram(fake_audio[:, :, :audio.size(2)].squeeze(1))
        loss_mel = l1loss(mel[:, :, :mel_fake.size(2)], mel_fake.cuda())
        loss_g += hp.model.lambda_mel * loss_mel


        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            adv_loss += criterion(score_fake, torch.ones_like(score_fake))


            if hp.model.feat_loss :
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d_real += criterion(score_real, torch.ones_like(score_real))
            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
        adv_loss = adv_loss / len(disc_fake)
        # MPD Adverserial loss
        adv_mpd_loss = 0

        mpd_fake_scores = model_d_mpd(fake_audio[:, :, :audio.size(2)])
        mpd_real_scores = model_d_mpd(audio)
        for score_fake, score_real in zip(mpd_fake_scores, mpd_real_scores):
            adv_mpd_loss += criterion(score_fake, torch.ones_like(score_fake))
            loss_mpd_real += criterion(score_real, torch.ones_like(score_real))
            loss_mpd_fake += criterion(score_fake, torch.zeros_like(score_fake))
        adv_mpd_loss = adv_mpd_loss / len(mpd_real_scores)
        adv_loss = adv_loss + adv_mpd_loss
        loss_mpd = (loss_mpd_fake + loss_mpd_real) / len(mpd_real_scores)


        loss_d_real = loss_d_real / len(score_real)
        loss_d_fake = loss_d_fake / len(disc_fake)
        loss_g += hp.model.lambda_adv * adv_loss
        loss_d = loss_d_real + loss_d_fake + loss_mpd
        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

        loader.set_description("g %.04f d %.04f ad %.04f| step %d" % (loss_g, loss_d, adv_loss, step))

    loss_g_avg = loss_g_sum / len(valloader.dataset)
    loss_d_avg = loss_d_sum / len(valloader.dataset)

    audio = audio[0][0].cpu().detach().numpy()
    fake_audio = fake_audio[0][0].cpu().detach().numpy()

    writer.log_validation(loss_g_avg, loss_d_avg, adv_loss, loss_mel.item(), loss_mpd.item(),\
                          generator, discriminator, audio, fake_audio, step)

    torch.backends.cudnn.benchmark = True
    generator.train()
    discriminator.train()
