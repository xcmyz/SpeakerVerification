import torch
import torch.nn as nn
from torch import optim

from network import SpeakerEncoder, SpeakerVerification
from data_utils import DataLoader, collate_fn
from data_utils import SVData
from loss import GE2ELoss
import hparams as hp

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model_SE = SpeakerEncoder().to(device)
    model_SV = SpeakerVerification().to(device)
    GE2E_loss = GE2ELoss()
    SV_loss = nn.CrossEntropyLoss()
    print("Models and Loss Have Been Defined")

    # Optimizer
    optimizer_SE = torch.optim.SGD([
        {'params': model_SE.parameters()},
        {'params': GE2E_loss.parameters()}],
        lr=hp.learning_rate
    )
    optimizer_SV = torch.optim.Adam(model_SV.parameters(), lr=1e-3)

    # Load checkpoint if exists
    try:
        checkpoint_SE = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_SE_%d.pth.tar' % args.restore_step))
        model_SE.load_state_dict(checkpoint_SE['model'])
        optimizer_SE.load_state_dict(checkpoint_SE['optimizer'])

        checkpoint_SV = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_SV_%d.pth.tar' % args.restore_step))
        model_SV.load_state_dict(checkpoint_SV['model'])
        optimizer_SV.load_state_dict(checkpoint_SV['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # # Change Learning Rate
    # learning_rate = 0.005
    # for param_group in optimizer_SE.param_groups:
    #     param_group['lr'] = learning_rate

    # Get dataset
    dataset = SVData()

    # Get training loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.N,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn,
                                 num_workers=cpu_count())

    # Define Some Information
    total_step = hp.epochs * len(training_loader)
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model_SE = model_SE.train()
    model_SV = model_SV.train()

    for epoch in range(hp.epochs):

        dataset = SVData()
        training_loader = DataLoader(dataset,
                                     batch_size=hp.N,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=collate_fn,
                                     num_workers=cpu_count())

        for i, batch in enumerate(training_loader):
            start_time = time.perf_counter()

            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            optimizer_SE.zero_grad()
            optimizer_SV.zero_grad()

            # Load Data
            mels, lengths, target = batch
            mels = torch.from_numpy(mels).float().to(device)
            target = torch.Tensor(target).long().to(device)

            # Forward
            speaker_embeddings = model_SE(mels, lengths)
            speaker_embeddings_ = torch.Tensor(
                speaker_embeddings.cpu().data).to(device)
            out = model_SV(speaker_embeddings_)

            # Loss
            speaker_embeddings = speaker_embeddings.contiguous().view(hp.N, hp.M, -1)
            ge2e_loss = GE2E_loss(speaker_embeddings)
            classify_loss = SV_loss(out, target)

            # Backward
            ge2e_loss.backward()
            classify_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model_SE.parameters(), 3.0)
            nn.utils.clip_grad_norm_(GE2E_loss.parameters(), 1.0)

            # Update weights
            optimizer_SE.step()
            optimizer_SV.step()

            if current_step % hp.log_step == 0:
                Now = time.perf_counter()
                str_loss = "Epoch [{}/{}], Step [{}/{}], GE2E Loss: {:.4f}, Classify Loss: {:.4f}.".format(
                    epoch + 1, hp.epochs, current_step, total_step, ge2e_loss.item(), classify_loss.item())
                str_time = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now - Start), (total_step - current_step) * np.mean(Time))

                print(str_loss)
                print(str_time)

                with open("logger.txt", "a")as f_logger:
                    f_logger.write(str_loss + "\n")
                    f_logger.write(str_time + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model_SE.state_dict(), 'optimizer': optimizer_SE.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_SE_%d.pth.tar' % current_step))
                torch.save({'model': model_SV.state_dict(), 'optimizer': optimizer_SV.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_SV_%d.pth.tar' % current_step))
                print("\nsave model at step %d ...\n" % current_step)

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
