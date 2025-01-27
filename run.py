import time
import random
import torch.cuda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.MsDL import *
from utils.Loss import Separation_Loss, Fitting_Loss
import torch.utils.data as Data
from sklearn.ensemble import RandomForestClassifier
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data(path):
    data_dict = np.load(path)
    train_x = torch.from_numpy(data_dict['train_x'].astype(np.float32))
    train_y = torch.from_numpy(data_dict['train_y'].astype(np.float32))
    test_x = torch.from_numpy(data_dict['test_x'].astype(np.float32))
    test_y = torch.from_numpy(data_dict['test_y'].astype(np.float32))
    num = train_x.shape[0]
    random_index = list(range(num))
    random.shuffle(random_index)
    train_x = train_x[random_index, :, :]
    train_y = train_y[random_index]
    return train_x, train_y, test_x, test_y



def transform(model, loader, radii, is_eval=False):
    model.is_eval = is_eval
    radii = radii.to(device)
    features, f_norms = [], []
    for x in loader:
        x = x[0].to(device)
        feature, f_norm = model(x, radii)
        features.append(feature)
        f_norms.append(f_norm)
    features = torch.cat(features, dim=0)
    if is_eval is False:
        f_norms = torch.sum(torch.stack(f_norms), dim=0)
    return features, f_norms


def train(model, x, y, batch_size, epsilon=0.01, p=4):
    assert p % 2 == 0, "p is preferably an even number."
    k = model.k
    num_samples = x.shape[0]
    if num_samples > 500:
        random_indices = torch.randperm(num_samples)[:500]
    else:
        random_indices = torch.arange(num_samples)
    loader = Data.DataLoader(dataset=Data.TensorDataset(x), batch_size=batch_size, shuffle=False, num_workers=0)
    s_loss = Separation_Loss(y[random_indices])
    f_loss = Fitting_Loss()
    left, right, width = torch.zeros(k), torch.ones(k), 1.

    best_radii = torch.ones(k)
    best_features, f_norms = transform(model, loader, best_radii, is_eval=False)
    best_loss = f_loss(best_features, f_norms) + s_loss(best_features[random_indices])

    radii = torch.ones(k) * 0.5
    features, f_norms = transform(model, loader, radii, is_eval=False)
    loss = f_loss(features, f_norms) + s_loss(features[random_indices])
    indices = torch.where(loss < best_loss)[0]
    best_radii[indices] = radii[indices]
    best_loss[indices] = loss[indices]
    best_features[:, indices, :] = features[:, indices, :]

    while width > epsilon * 2:
        delta = width / p
        pending = list(range(1, p//2)) + list(range(p//2 + 1, p))
        for i in pending:
            radii = left + (delta * i)
            features, f_norms = transform(model, loader, radii, is_eval=False)
            loss = f_loss(features, f_norms) + s_loss(features[random_indices])
            indices = torch.where(loss < best_loss)[0]
            best_radii[indices] = radii[indices]
            best_loss[indices] = loss[indices]
            best_features[:, indices, :] = features[:, indices, :]
        right_indices = torch.where(best_radii == 1.)[0]
        mid_indices = torch.where(best_radii < 1.)[0]
        left[right_indices] = 1. - 2 * delta
        right[right_indices] = 1.
        left[mid_indices] = best_radii[mid_indices] - delta
        right[mid_indices] = best_radii[mid_indices] + delta
        width = 2 * delta

    return best_radii, best_features, best_loss


def run(units=20, regular=1., leaky=0., K=40):
    set_seed(42)
    x_train, y_train, x_test, y_test = get_data("./data/ECGFiveDays.npz")
    train_batch = x_train.shape[0]
    test_batch = x_test.shape[0]
    input_dim = x_train.shape[2]
    te_loader = Data.DataLoader(dataset=Data.TensorDataset(x_test),
                                batch_size=test_batch, shuffle=False, num_workers=0)
    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    if device == "cpu":
        model = MsDL_CPU(input_dim, k=min(x_train.shape[1] - 1, K), num_units=units,
                         leaky=leaky, regular=regular)
    else:
        model = MsDL_GPU(input_dim, k=min(x_train.shape[1] - 1, K), num_units=units,
                         leaky=leaky, regular=regular)
    model = model.to(device)

    start = time.time()
    best_radii, train_features, _ = train(model, x_train, y_train, batch_size=train_batch)
    train_features = train_features.flatten(start_dim=1)
    y_train, y_test = y_train.numpy(), y_test.numpy()
    clf.fit(train_features, y_train)
    mid = time.time()
    test_features, _ = transform(model, te_loader, best_radii, is_eval=True)
    test_features = test_features.flatten(start_dim=1)
    y_pred = clf.predict(test_features)
    end = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    result = [accuracy, precision, recall, f1, mid - start, end - mid]
    print(result)
    return result


if __name__ == "__main__":
    run()










