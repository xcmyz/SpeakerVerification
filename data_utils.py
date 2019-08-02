import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from multiprocessing import cpu_count
from nnmnkwii.datasets import vctk

import hparams


class SVData(Dataset):
    """ VCTK """

    def __init__(self, dataset_path=hparams.dataset_path):
        self.dataset_path = dataset_path

    def __len__(self):
        return len(vctk.available_speakers)

    def random_sample(self, total_length, sample_length):
        out = [i for i in range(total_length)]
        return random.sample(out, sample_length)

    def __getitem__(self, index):
        id_list = self.random_sample(
            len(os.listdir(os.path.join(self.dataset_path, str(index)))), hparams.M)

        list_file_name = list()
        for i in id_list:
            file_name = str(index) + "_" + str(i) + ".npy"
            list_file_name.append(os.path.join(os.path.join(
                self.dataset_path, str(index)), file_name))

        return [np.load(file_name).T for file_name in list_file_name], index


def collate_fn(batch):
    lengths = list()
    index_list = list()
    mel_all_list = list()

    for mels, index in batch:
        index_list.append(index)
        for mel in mels:
            lengths.append(mel.shape[0])
            mel_all_list.append(mel)
    max_len = max(lengths)
    mels = pad(mel_all_list, max_len)

    target = list()
    for ele in index_list:
        for _ in range(hparams.M):
            target.append(ele)

    return mels, lengths, target


def pad(mels, max_len):
    mel_list = list()
    for mel in mels:
        pad_size = ((0, max_len-mel.shape[0]), (0, 0))
        mel_list.append(
            np.pad(mel, pad_size, mode='constant', constant_values=0))
    mels_padded = np.stack(mel_list)

    return mels_padded


if __name__ == "__main__":
    # Test
    test_dataloader = DataLoader(SVData(),
                                 batch_size=5,
                                 shuffle=True,
                                 drop_last=False,
                                 collate_fn=collate_fn,
                                 num_workers=cpu_count())

    for i, data in enumerate(test_dataloader):
        mels, lengths, label = data
        print(mels.shape)
        print(label)
        # print(len(lengths))
        # print(lengths)
