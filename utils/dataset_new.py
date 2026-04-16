import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def ComFold(batchsize, Filename, nfold, fold):
    data = np.genfromtxt(Filename[0], delimiter=',')
    label = np.genfromtxt(Filename[1], delimiter=',')
    com_label = np.genfromtxt(Filename[2], delimiter=',')
    n_test = len(com_label)//nfold
    print('n size:', n_test)
    y = np.arange(len(com_label))
    start = fold*n_test
    if start+n_test > len(com_label):
        test = y[start:]
    else:
        test = y[start:start+n_test]
    train = np.setdiff1d(y, test)

    # training dataset
    train_scene = ComFoldData(data[train, :], label[train, :], com_label=com_label[train, :],disam_com_label=None, train=True)
    print("train data shape:", data[train, :].shape)
    train_loader = DataLoader(
        train_scene,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    # Data loader for test dataset
    size = len(test)
    test_scene = ComFoldData(data[test, :], label[test, :], com_label=com_label[test, :],disam_com_label=None, train=False)
    print("test data & label shape:", data[test, :].shape, label[test, :].shape)
    test_loader = DataLoader(
        test_scene,
        batch_size=size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, test_loader
class ComFoldData(Dataset):
    def __init__(self, data, label, com_label, disam_com_label,finalF=None,train=True):
        # reading data as table form
        self.images = data
        # reading labels
        self.labels = label

        self.comp_labels = com_label
        self.disam_com_label=disam_com_label
        # initial confidence
        if finalF is None:
            self.finalF=None
        else:
            self.finalF = finalF.detach().numpy()
        self.train = train

    def __getitem__(self, index):
        img, target, comp_label = torch.from_numpy(self.images[index]).float(), \
                                              torch.from_numpy(self.labels[index]).float(), \
                                              torch.from_numpy(self.comp_labels[index]).float()
        if self.disam_com_label is None:
            disam_com_label =0
        else:
            disam_com_label = torch.from_numpy(self.disam_com_label[index]).float()
        if self.finalF is None:
            finalF =0
        else:
            finalF = torch.from_numpy(self.finalF[index]).float()

        return img, target, comp_label,disam_com_label, finalF, index

    def __len__(self):
        return len(self.labels)