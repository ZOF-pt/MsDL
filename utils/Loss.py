import torch
import torch.nn as nn


class Separation_Loss(nn.Module):
    def __init__(self, labels, sigma=1.):
        super(Separation_Loss, self).__init__()
        self.sigma = sigma
        unique_labels, counts = labels.unique(return_counts=True)
        self.counts = torch.zeros_like(labels)
        for label, count in zip(unique_labels, counts):
            self.counts[labels == label] = 1 / count

        row_indices = []
        col_indices = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] == labels[j]:
                    row_indices.append(i)
                    col_indices.append(j)
        self.row_indices = torch.tensor(row_indices)
        self.col_indices = torch.tensor(col_indices)

    def forward(self, F):
        F = F / (pow(2, 0.5) * self.sigma)
        F_i = F.unsqueeze(1)
        F_j = F.unsqueeze(0)
        dis = (F_i - F_j) ** 2
        dis = dis.sum(dim=-1)
        sim = torch.exp(-dis)
        p1 = sim.sum(dim=1) * self.counts.unsqueeze(1)
        p1 = torch.sum(torch.log(p1), dim=0)
        p2 = torch.log(sim[self.row_indices, self.col_indices, :])
        p2 = torch.sum(p2 * self.counts[self.row_indices].unsqueeze(1), dim=0)
        return p1 - p2


class Fitting_Loss(nn.Module):
    def __init__(self, sigma=1.):
        super(Fitting_Loss, self).__init__()
        self.sigma = sigma

    def forward(self, F, Norm):
        return Norm + self.sigma * torch.sum(F ** 2, dim=[0, 2])
