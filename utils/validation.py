import tqdm
import torch


def validate(hp, generator, discriminator, model_d_mpd, valloader, stft_loss, l1loss, criterion, stft, writer, step):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for mel, audio in loader:
        mel = mel.cuda()
        audio = audio.cuda()    # B, 1, T torch.Size([1, 1, 212893])

        # generator
        fake_audio = generator(mel) # B, 1, T' torch.Size([1, 1, 212992])
        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)]) # B, 1, T torch.Size([1, 1, 212893])
        disc_real = discriminator(audio)

        adv_loss =0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audio.size(2)].squeeze(1), audio.squeeze(1))
        loss_g = sc_loss + mag_loss

        # Mel Loss

        mel_fake = stft.mel_spectrogram(fake_audio[:, :, :audio.size(2)].squeeze(1))
        loss_mel = l1loss(mel[:, :, :mel_fake.size(2)], mel_fake.cuda())
        loss_g += hp.model.lambda_mel * loss_mel


        for (score_fake, feats_fake), (score_real, feats_real) in zip(disc_fake, disc_real):
            adv_loss += criterion(score_fake, torch.ones_like(score_fake))


            if hp.model.feat_loss :
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d_real += criterion(score_real, torch.ones_like(score_real))
            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
        adv_loss = adv_loss / len(disc_fake)

        # MPD Adverserial loss
        mpd_fake = model_d_mpd(fake_audio[:, :, :audio.size(2)])
        mpd_real = model_d_mpd(audio)
        for score_fake, feats_fake in mpd_fake:
            adv_mpd_loss = criterion(score_fake, torch.ones_like(score_fake))
        adv_mpd_loss = adv_mpd_loss / len(mpd_fake)

        if hp.model.feat_loss:
            for (_, feats_fake), (_, feats_real) in zip(mpd_fake, mpd_real):
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

        adv_loss = adv_loss + adv_mpd_loss

        for (score_fake, _), (score_real, _) in zip(mpd_fake, mpd_real):
            loss_mpd_real = criterion(score_real, torch.ones_like(score_real))
            loss_mpd_fake = criterion(score_fake, torch.zeros_like(score_fake))
        loss_mpd = (loss_mpd_fake + loss_mpd_real) / len(mpd_real)  # MPD Loss


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