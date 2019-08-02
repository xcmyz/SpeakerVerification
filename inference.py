import torch
import numpy as np
import random
import os

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from network import SpeakerEncoder, SpeakerVerification
from data_utils import DataLoader, SVData, collate_fn
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pca(embeddings, dim=3):
    pac_model = PCA(dim)
    pca_embeddings = pac_model.fit_transform(embeddings)

    return pca_embeddings


def draw_pic_3D(embeddings, speaker_len, utter_len):
    pca_embeddings = pca(embeddings)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, speaker_len)]
    ax = plt.subplot(111, projection='3d')

    for ind, ele in enumerate(pca_embeddings):
        ax.scatter(ele[0], ele[1], ele[2], color=colors[ind // utter_len])

    plt.suptitle("RESULT")
    plt.title("Speaker Embedding")

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.savefig("3d.jpg")


def draw_pic_2D(embeddings, speaker_len, utter_len):
    pca_embeddings = pca(embeddings, 2)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, speaker_len)]

    plt.figure()
    for ind, ele in enumerate(pca_embeddings):
        plt.scatter(ele[0], ele[1], color=colors[ind // utter_len])
    plt.savefig("2d.jpg")


def test(model_SE, model_SV):
    file_name_list = list()
    targets = list()
    for i in range(108):
        list_file = os.listdir(os.path.join(hp.dataset_path, str(i)))
        length_file = len(list_file)
        for _ in range(200):
            index = random.randint(0, length_file-1)
            file_name_list.append(os.path.join(os.path.join(
                hp.dataset_path, str(i)), list_file[index]))
            targets.append(i)

    speaker_embeddings = list()
    cnt = 0
    with torch.no_grad():
        for i, file_name in enumerate(file_name_list):
            mel = torch.from_numpy(np.load(file_name).T).float().to(
                device).unsqueeze(0)
            length = [np.load(file_name).T.shape[0]]

            speaker_embedding = model_SE(mel, length)
            out = model_SV(speaker_embedding)

            speaker_embeddings.append(speaker_embedding[0].cpu().numpy())

            _, predicted = torch.max(out.data, 1)

            if predicted.data == targets[i]:
                cnt += 1

            if (i+1) % 100 == 0:
                print("Done", (i+1))

    speaker_embeddings = torch.Tensor(speaker_embeddings).numpy()
    return cnt / len(targets), speaker_embeddings


if __name__ == "__main__":

    # Define model
    model_SE = SpeakerEncoder().to(device)
    model_SV = SpeakerVerification().to(device)
    model_SE.eval()
    model_SV.eval()
    print("Model Have Been Defined")

    # Load checkpoint
    num = 35000
    checkpoint_SE = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_SE_' + str(num) + '.pth.tar'))
    checkpoint_SV = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_SV_' + str(num) + '.pth.tar'))
    model_SE.load_state_dict(checkpoint_SE['model'])
    model_SV.load_state_dict(checkpoint_SV['model'])
    print("Load Done")

    # Test
    acc, speaker_embeddings = test(model_SE, model_SV)
    print("\nThe accuracy of this Speaker Verification model is {:.4f}%.".format(
        acc * 100))

    # draw_pic_2D(speaker_embeddings, 10, 20)
    # draw_pic_3D(speaker_embeddings, 10, 20)
