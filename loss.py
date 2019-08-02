import torch
import torch.nn as nn

import utils
import hparams as hp


device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class GE2ELoss(nn.Module):
    """GE2E Loss"""

    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, embeddings):
        torch.clamp(self.w, hp.re_num)

        centroids = utils.get_centroids(embeddings)
        cossim = utils.get_cossim(embeddings, centroids)

        sim_matrix = self.w * cossim + self.b
        loss, _ = utils.cal_loss(sim_matrix)

        return loss


if __name__ == "__main__":

    loss = GE2ELoss()
    print(loss)
