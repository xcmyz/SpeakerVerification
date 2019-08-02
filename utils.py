import torch
import torch.autograd as grad
import torch.nn.functional as F

import librosa
import random
import numpy as np

import hparams as hp


def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid / speaker.size(0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)

    return centroids


def get_centroid_remove(embeddings, speaker_num, utterance_num):
    # makes training stable and helps avoid trivial solutions

    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid / (embeddings[speaker_num].size(0) - 1)

    return centroid


def get_cossim(embeddings, centroids):
    # Calculates cosine similarity matrix
    # Requires (N, M, feature) input

    cossim = torch.zeros(embeddings.size(
        0), embeddings.size(1), centroids.size(0))

    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):

                if speaker_num == centroid_num:
                    centroid = get_centroid_remove(
                        embeddings, speaker_num, utterance_num)

                output = F.cosine_similarity(
                    utterance, centroid, dim=0) + hp.re_num
                cossim[speaker_num][utterance_num][centroid_num] = output

    return cossim


def cal_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix

    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))

    for j in range(sim_matrix.size(0)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -sim_matrix[j][i][j] + \
                ((torch.exp(sim_matrix[j][i]).sum() + hp.re_num).log_())

    loss = per_embedding_loss.sum()

    return loss, per_embedding_loss


def cutstr(string):
    index = 0
    for ind, char in enumerate(string):
        if char == "_":
            index = ind
            break
    i = 0
    for ind, char in enumerate(string):
        if char == ".":
            i = ind
            break
    first = int(string[0:index])
    end = int(string[index + 1:i])
    return first, end


def random_sample(list_for_sample, num):
    index_list = random.sample([i for i in range(len(list_for_sample))], num)
    return [list_for_sample[ind] for ind in index_list]


def random_cut(mel, length=hp.tisv_frame):
    total_length = np.shape(mel)[0]

    if total_length <= hp.tisv_frame:
        # raise ValueError("total length is too short!")
        mel = np.concatenate((mel, mel[0:total_length, :]))

    total_length = np.shape(mel)[0]

    if total_length < 181:
        # raise ValueError("something wrong!")
        mel = np.concatenate((mel, mel[0:total_length, :]))

    total_length = np.shape(mel)[0]

    if total_length < 181:
        # raise ValueError("something wrong!")
        mel = np.concatenate((mel, mel[0:total_length, :]))

    total_length = np.shape(mel)[0]

    start = random.randint(0, total_length - length - 1)

    return mel[start:start + length, :]


if __name__ == "__main__":

    test_matrix = torch.randn(20, 30, 6)
    # test_matrix = torch.randn(2, 20, 30, 6)
    # print(test_matrix)

    centroids = get_centroids(test_matrix)
    print(centroids.size())

    sim = get_cossim(test_matrix, centroids)
    print(sim.size())

    loss, _ = cal_loss(sim)
    print(loss)

    mel = np.ndarray((180, 40))
    print(np.shape(random_cut(mel, 180)))
