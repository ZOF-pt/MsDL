import torch
import torch.nn as nn
import numpy as np


class MsDL_GPU(nn.Module):
    def __init__(self, input_dim, k=40, num_units=20, regular=1., leaky=0., uni_res=True, is_eval=False):
        super(MsDL_GPU, self).__init__()
        self.input_dim = input_dim
        self.k = k
        self.is_eval = is_eval
        self.leaky = leaky
        self.num_units = num_units
        self.connectivity = min(1., 10 / num_units)
        self.regular = nn.Parameter(regular * torch.eye(self.num_units).unsqueeze(0), requires_grad=False)
        self.W_in = nn.Parameter(torch.randn(input_dim, num_units), requires_grad=False)
        if uni_res is True:
            res = torch.from_numpy(self.get_res(num_units, self.connectivity, k=1).astype(np.float32))
            res = res.repeat(k, 1, 1)
        else:
            res = torch.from_numpy(self.get_res(num_units, self.connectivity, k=k).astype(np.float32))
        self.W_res = nn.Parameter(res, requires_grad=False)

    def get_res(self, hidden_dim, connectivity, k=1):
        w = np.random.randn(k, hidden_dim, hidden_dim)
        mask = np.random.choice([0, 1], size=(k, hidden_dim, hidden_dim),
                                p=[1 - connectivity, connectivity])
        w = w * mask
        for i in range(k):
            max_eigenvalue = max(abs(np.linalg.eigvals(w[i])))
            w[i] = w[i] / max_eigenvalue
        return w

    def compute_state(self, x_transformed, radii):
        batch_size, length, _ = x_transformed.size()
        x_transformed = x_transformed.unsqueeze(1)
        h_history = torch.zeros(batch_size, self.k, length-1, self.num_units, device=x_transformed.device)
        W_res = self.W_res * radii.view(self.k, 1, 1)
        for t in range(length-1):
            row_indices = torch.arange(min(self.k, length-t-1))
            col_indices = t - row_indices - 1
            h_pre = h_history[:, row_indices, col_indices, :]
            h_t = x_transformed[:, :, t, :] + torch.matmul(h_pre.unsqueeze(2), W_res[row_indices]).squeeze(2)
            h_history[:, row_indices, t, :] = self.leaky * h_pre + (1 - self.leaky) * torch.tanh(h_t)
        return h_history

    def forward(self, x, radii):
        batch_size, length, _ = x.size()
        assert self.k < length, f"Failed: K ({self.k}) should be less than length ({length})."
        x_transformed = torch.matmul(x, self.W_in)
        h_history = self.compute_state(x_transformed, radii)
        features, f_norms = [], []
        for i in range(self.k):
            H = h_history[:, i, :length - i - 1, :]
            X = x[:, i + 1:, :]
            Ht = H.transpose(1, 2)
            HtH = torch.bmm(Ht, H)
            HtX = torch.bmm(Ht, X)
            W_out = torch.linalg.solve(HtH + self.regular, HtX)
            if self.is_eval is False:
                error = torch.bmm(H, W_out) - X
                f_norm = torch.sum(error)
                f_norms.append(f_norm)
            features.append(W_out.unsqueeze(1))
        features = torch.cat(features, dim=1).flatten(start_dim=2).cpu()
        f_norms = torch.tensor(f_norms).cpu()
        return features, f_norms


class MsDL_CPU(nn.Module):
    def __init__(self, input_dim, k=40, num_units=20, regular=1., leaky=0., uni_res=True, is_eval=False):
        super(MsDL_CPU, self).__init__()
        self.input_dim = input_dim
        self.k = k
        self.is_eval = is_eval
        self.leaky = leaky
        self.num_units = num_units
        self.connectivity = min(1., 10 / num_units)
        self.regular = nn.Parameter(regular * torch.eye(self.num_units).unsqueeze(0), requires_grad=False)
        self.W_in = nn.Parameter(torch.randn(input_dim, num_units), requires_grad=False)
        if uni_res is True:
            res = torch.from_numpy(self.get_res(num_units, self.connectivity, k=1).astype(np.float32))
            res = res.repeat(k, 1, 1)
        else:
            res = torch.from_numpy(self.get_res(num_units, self.connectivity, k=k).astype(np.float32))
        self.W_res = nn.Parameter(res, requires_grad=False)

    def get_res(self, hidden_dim, connectivity, k=1):
        w = np.random.randn(k, hidden_dim, hidden_dim)
        mask = np.random.choice([0, 1], size=(k, hidden_dim, hidden_dim),
                                p=[1 - connectivity, connectivity])
        w = w * mask
        for i in range(k):
            max_eigenvalue = max(abs(np.linalg.eigvals(w[i])))
            w[i] = w[i] / max_eigenvalue
        return w

    def compute_state(self, x_transformed, skip, radius):
        # skip: 0:k
        batch_size, length, _ = x_transformed.size()
        h_history = torch.zeros(batch_size, length-1, self.num_units, device=x_transformed.device)
        for t in range(length-skip-1):
            h_pre = h_history[:, t-skip-1, :]
            h_t = x_transformed[:, t, :] + h_pre @ self.W_res[skip] * radius
            h_history[:, t, :] = self.leaky * h_pre + (1 - self.leaky) * torch.tanh(h_t)
        return h_history

    def forward(self, x, radii):
        batch_size, length, _ = x.size()
        assert self.k < length, f"Failed: K ({self.k}) should be less than length ({length})."
        x_transformed = torch.matmul(x, self.W_in)
        features, f_norms = [], []
        for skip in range(self.k):
            h_history = self.compute_state(x_transformed, skip, radii[skip])
            H = h_history[:, :length - skip - 1, :]
            X = x[:, skip + 1:, :]
            Ht = H.transpose(1, 2)
            HtH = torch.bmm(Ht, H)
            HtX = torch.bmm(Ht, X)
            W_out = torch.linalg.solve(HtH + self.regular, HtX)
            if self.is_eval is False:
                error = torch.bmm(H, W_out) - X
                f_norm = torch.sum(error)
                f_norms.append(f_norm)
            features.append(W_out.unsqueeze(1))
        features = torch.cat(features, dim=1).flatten(start_dim=2).cpu()
        f_norms = torch.tensor(f_norms).cpu()
        return features, f_norms


