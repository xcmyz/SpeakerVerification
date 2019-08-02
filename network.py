import torch
import torch.nn as nn

import hparams as hp


class SpeakerEncoder(nn.Module):
    """ Speaker Encoder """

    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        self.lstm = nn.LSTM(hp.n_mels_channel,
                            hp.hidden_dim,
                            num_layers=hp.num_layer,
                            batch_first=True)
        self.projection = nn.Linear(hp.hidden_dim, hp.speaker_dim)
        self.init_params()

    def init_params(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, input_lengths):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        out = list()
        for i in range(x.size(0)):
            out.append(x[i][input_lengths[i]-1])
        out = torch.stack(out)
        out = self.projection(out)
        out = out / torch.norm(out)

        return out


class SpeakerVerification(nn.Module):
    """ Speaker Verification """

    def __init__(self):
        super(SpeakerVerification, self).__init__()
        self.classify_net_1 = nn.Linear(hp.speaker_dim, hp.hidden_dim)
        self.classify_net_2 = nn.Linear(hp.hidden_dim, hp.class_num)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(
            self.classify_net_1.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, speaker_embeddings):
        out = self.classify_net_1(speaker_embeddings)
        out = self.relu(out)
        out = self.classify_net_2(out)

        return out
