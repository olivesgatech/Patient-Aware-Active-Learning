import numpy as np
import argparse
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import ntpath
import glob
import os


# Dataset code from https://github.com/JordanAsh/badge/blob/master/dataset.py
def get_dataset(name, path, download=True, corruption='fog', eval=None):
    if name == 'MNIST':
        return get_MNIST(path, download)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path, download)
    elif name == 'SVHN':
        return get_SVHN(path, download)
    elif name == 'CIFAR10':
        return get_CIFAR10(path, download, corruption)
    elif name == 'STL10':
        return get_STL10(path, download)
    elif name == 'Kermany':
        return get_Kermany(eval=eval)
    elif name == 'KermanyXray':
        return get_KermanyXray(eval=eval)


def extract_dataset_indices(image_paths, extract_list):
    # convert image list to file list
    image_list = [ntpath.basename(path)[:-4] for path in image_paths]
    out = []
    # get all indices
    for image in extract_list:
        index = image_list.index(image)
        out.append(index)

    return out


def get_Kermany(eval=None):
    file_path = './Data/Kermany/'

    raw_tr = np.load(os.path.join(file_path, 'train.npy'), allow_pickle=True)
    raw_te = np.load(os.path.join(file_path, 'test.npy'), allow_pickle=True)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr[:, 0]
    output_dict['Y_tr'] = raw_tr[:, 1]
    output_dict['X_te'] = raw_te[:, 0]
    output_dict['Y_te'] = raw_te[:, 1]
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 3
    output_dict['Y_te_ID'] = np.squeeze(raw_te[:, 2])  # patient ID in test set
    output_dict['Y_tr_ID'] = np.squeeze(raw_tr[:, 2])  # patient ID in train set
    output_dict['eval'] = eval
    return output_dict


def get_KermanyXray(eval=None):
    file_path = './Data/Kermany_xray/'

    raw_tr = np.load(os.path.join(file_path, 'train.npy'), allow_pickle=True)
    raw_te = np.load(os.path.join(file_path, 'test.npy'), allow_pickle=True)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr[:, 0]
    output_dict['Y_tr'] = raw_tr[:, 1]
    output_dict['X_te'] = raw_te[:, 0]
    output_dict['Y_te'] = raw_te[:, 1]
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 3
    output_dict['Y_te_ID'] = np.squeeze(raw_te[:, 2])  # patient ID in test set
    output_dict['Y_tr_ID'] = np.squeeze(raw_tr[:, 2])  # patient ID in train set
    output_dict['eval'] = eval
    return output_dict


def get_STL10(path, download=True):
    raw_tr = datasets.STL10(path + '/STL10', split='train', download=download)
    raw_te = datasets.STL10(path + '/STL10', split='test', download=download)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr.data
    output_dict['Y_tr'] = raw_tr.labels
    output_dict['X_te'] = raw_te.data
    output_dict['Y_te'] = raw_te.labels
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 10
    return output_dict


def get_MNIST(path, download=True):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=download)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=download)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr.train_data
    output_dict['Y_tr'] = raw_tr.train_labels
    output_dict['X_te'] = raw_te.test_data
    output_dict['Y_te'] = raw_te.test_labels
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 10
    return output_dict


def get_FashionMNIST(path, download=True):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=download)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=download)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr.train_data
    output_dict['Y_tr'] = raw_tr.train_labels
    output_dict['X_te'] = raw_te.test_data
    output_dict['Y_te'] = raw_te.test_labels
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 10
    return output_dict


def get_SVHN(path, download=True):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=download)
    data_te = datasets.SVHN(path + '/SVHN', split='test', download=download)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = data_tr.data
    output_dict['Y_tr'] = torch.from_numpy(data_tr.labels)
    output_dict['X_te'] = data_te.data
    output_dict['Y_te'] = torch.from_numpy(data_te.labels)
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 10
    return output_dict


def get_CIFAR10(path, download=True, corruption='fog'):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=download)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=download)

    # init output dict
    output_dict = {}
    if corruption is not None:
        x_corrupted_te = np.load(path + '/CIFAR-10-C/' + corruption + '.npy')
        y_corrupted_te = np.load(path + '/CIFAR-10-C/labels.npy')
        output_dict['X_corr_te'] = x_corrupted_te[30000:]
        output_dict['Y_corr_te'] = torch.from_numpy(y_corrupted_te[30000:])
    output_dict['X_tr'] = data_tr.data
    output_dict['Y_tr'] = torch.from_numpy(np.array(data_tr.targets))
    output_dict['X_te'] = data_te.data
    output_dict['Y_te'] = torch.from_numpy(np.array(data_te.targets))
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['nclasses'] = 10
    return output_dict


def make_data_loader(args, current_idxs=None, **kwargs):
    # define augmentations
    # Setup train transforms
    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomHorizontalFlip())

    # different transforms for every dataset
    if args.dataset == 'MNIST':
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        train_transform.transforms.append(transforms.RandomCrop(28, padding=3))
        train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
        img_size = 28
    elif args.dataset == 'CIFAR10':
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        # train_transform.transforms.append(transforms.RandomRotation(10))
        # train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        img_size = 32
    elif args.dataset == 'PASCALVOC':
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
        train_transform.transforms.append(transforms.Resize((300, 300)))
        train_transform.transforms.append(transforms.RandomChoice([transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                                   transforms.RandomGrayscale(p=0.25)]))
        train_transform.transforms.append(transforms.RandomRotation(25))
    elif args.dataset == 'STL10':
        mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
        std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
        train_transform.transforms.append(transforms.RandomCrop(96, padding=4))
        # train_transform.transforms.append(transforms.RandomChoice([transforms.ColorJitter(brightness=(0.80, 1.20)),
        #                                                           transforms.RandomGrayscale(p=0.25)]))
        # train_transform.transforms.append(transforms.RandomRotation(10))
        img_size = 96
    elif args.dataset == 'SVHN':
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        # train_transform.transforms.append(transforms.RandomRotation(10))
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        img_size = 32
    elif args.dataset == 'Kermany':
        mean = 0.19876538343581782
        std = 0.07863086417936489
        train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
        img_size = 128
    elif args.dataset == 'KermanyXray':
        mean = 0.48232842642260854
        std = 0.037963852433605165
        train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
        img_size = 128
    else:
        raise NotImplementedError
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # Setup test transforms

    # add multiplex to several channels with mnist
    if args.dataset == 'MNIST':
        test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
    elif args.dataset == 'PASCALVOC':
        test_transform = transforms.Compose([transforms.Resize(330), transforms.CenterCrop(300), transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
    else:
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # get corresponding dataset
    trainhandler = get_handler(args.dataset)
    testhandler = get_handler(args.dataset)
    data_configs = get_dataset(args.dataset, args.data_path, args.download, args.corruption, eval=args.eval)

    # init loaders
    if current_idxs is not None:
        train_loader = DataLoader(trainhandler(data_dict=data_configs, current_idxs=current_idxs, split='tr',
                                               transform=train_transform),
                                  batch_size=args.batch_size, shuffle=True, **kwargs)

        # train loader for gradient embeddings
        total_idxs = np.arange(data_configs['tr_len'])
        unused_idxs = np.delete(total_idxs, current_idxs)
        # split must be tr because we are sampling from unused training data not test
        grad_loader = DataLoader(trainhandler(data_dict=data_configs, current_idxs=unused_idxs, split='tr',
                                              transform=test_transform), batch_size=1, shuffle=True, **kwargs)
        unlabeled_loader = DataLoader(trainhandler(data_dict=data_configs, current_idxs=unused_idxs, split='tr',
                                                   transform=test_transform), batch_size=args.unlabeled_batch_size,
                                      shuffle=True, **kwargs)
    else:
        train_loader = DataLoader(trainhandler(data_dict=data_configs, transform=train_transform, split='tr'),
                                  batch_size=args.batch_size, shuffle=True, **kwargs)

        # train loader for gradient embeddings
        grad_loader = DataLoader(trainhandler(data_dict=data_configs, transform=test_transform, split='tr'),
                                 batch_size=1, shuffle=False, **kwargs)
        unlabeled_loader = DataLoader(trainhandler(data_dict=data_configs, transform=test_transform, split='tr'),
                                      batch_size=args.unlabeled_batch_size, shuffle=True, **kwargs)
    # test data loader
    test_loader = DataLoader(testhandler(data_dict=data_configs, transform=test_transform, split='te'),
                             batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # add corrupted test loader if available
    if 'X_corr_te' in data_configs.keys():
        corr_loader = DataLoader(testhandler(data_dict=data_configs, transform=test_transform, split='corr_te'),
                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        corr_loader = None

    # get dataset size
    train_pool = data_configs['tr_len']
    test_pool = data_configs['te_len']

    # write output to dict
    output = {}
    output['train_pool'] = train_pool
    output['test_pool'] = test_pool
    output['test_loader'] = test_loader
    output['train_loader'] = train_loader
    output['grad_loader'] = grad_loader
    output['unlabeled_loader'] = unlabeled_loader
    output['corrupted_loader'] = corr_loader
    output['nclasses'] = data_configs['nclasses']
    # if args.dataset == 'Kermany':
    #     output['dim'] = img_size*img_size
    # else:
    output['dim'] = img_size*img_size*3
    return output


def get_handler(name):
    if name == 'MNIST':
        return LoaderMNIST
    elif name == 'FashionMNIST':
        return LoaderMNIST
    elif name == 'SVHN':
        return LoaderSVHN
    elif name == 'CIFAR10':
        return LoaderCIFAR10
    elif name == 'STL10':
        return LoaderSTL10
    elif name == 'Kermany':
        return LoaderKermany
    elif name == 'KermanyXray':
        return LoaderKermanyXray
    else:
        return DataHandler4


class LoaderKermany(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')
        self.split = split
        self.current_idxs = current_idxs
        self.ID = None
        self.eval = data_dict['eval']
        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
            if split == 'te':
                self.ID = np.squeeze(data_dict['Y_te_ID'][current_idxs])  # the patient ID's in the test set
            else:
                self.ID = np.squeeze(data_dict['Y_tr_ID'][current_idxs])  # the patient ID's in the train set
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
            if split == 'te':
                self.ID = np.squeeze(data_dict['Y_te_ID'][current_idxs])
            else:
                self.ID = np.squeeze(data_dict['Y_tr_ID'][current_idxs])
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.split == 'te':
            id = self.ID[index]
        else:
            id = self.ID[index]

        if len(x.shape) == 3:  # happens a few times in test set
            x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        if len(x.shape) != 3:
            x = np.stack((x,)*3, axis=0)
        y -= 1  # make labels 0, 1, 2 instead of 1, 2, 3

        if self.transform is not None:
            permuted = np.moveaxis(x, [0, 1, 2], [2, 0, 1])
            im = Image.fromarray(permuted)
            x = self.transform(im)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        if self.eval == 'Patient':
            return x, y, out_index, id
        else:
            return x, y, out_index

    def __len__(self):
        return len(self.X)


class LoaderKermanyXray(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')
        self.split = split
        self.current_idxs = current_idxs
        self.ID = None
        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
            if split == 'te':
                self.ID = np.squeeze(data_dict['Y_te_ID'][current_idxs])  # the patient ID's in the test set
            else:
                self.ID = np.squeeze(data_dict['Y_tr_ID'][current_idxs])  # the patient ID's in the train set
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
            if split == 'te':
                self.ID = np.squeeze(data_dict['Y_te_ID'][current_idxs])
            else:
                self.ID = np.squeeze(data_dict['Y_tr_ID'][current_idxs])
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.split == 'te':
            id = self.ID[index]
        else:
            id = self.ID[index]

        if len(x.shape) == 3:  # happens a few times in test set
            x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        if len(x.shape) != 3:
            x = np.stack((x,)*3, axis=0)
        # y -= 1  # make labels 0, 1, 2 instead of 1, 2, 3

        if self.transform is not None:
            permuted = np.moveaxis(x, [0, 1, 2], [2, 0, 1])
            im = Image.fromarray(permuted)
            x = self.transform(im)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        # if self.split == 'te':
        return x, y, out_index, id
        # else:
        #     return x, y, out_index

    def __len__(self):
        return len(self.X)


class LoaderSTL10(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        # make sure split is in correct format
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')

        self.current_idxs = current_idxs

        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            permuted = np.moveaxis(x, [0, 1, 2], [2, 0, 1])
            im = Image.fromarray(permuted)
            x = self.transform(im)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        return x, y, out_index

    def __len__(self):
        return len(self.X)


class LoaderMNIST(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        # make sure split is in correct format
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')

        self.current_idxs = current_idxs

        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        return x, y, out_index

    def __len__(self):
        return len(self.X)


class LoaderSVHN(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        # make sure split is in correct format
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')

        self.current_idxs = current_idxs

        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        return x, y, out_index

    def __len__(self):
        return len(self.X)


class LoaderCIFAR10(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        # make sure split is in correct format
        if split != 'tr' and split != 'te' and split != 'corr_te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or '
                            '\'te\' or \'corr_te\'!!!')

        self.current_idxs = current_idxs

        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            im = Image.fromarray(x)
            x = self.transform(im)

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        return x, y, out_index

    def __len__(self):
        return len(self.X)


class DataHandler4(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        # make sure split is in correct format
        if split != 'tr' and split != 'te':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')

        self.current_idxs = current_idxs

        # initialize data
        if current_idxs is not None:
            self.X = data_dict['X_' + split][current_idxs]
            self.Y = data_dict['Y_' + split][current_idxs]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]

        if self.current_idxs is not None:
            out_index = self.current_idxs[index]
        else:
            out_index = index
        return x, y, out_index

    def __len__(self):
        return len(self.X)

# for testing
def parse_everything():
    parser = argparse.ArgumentParser(description="PyTorch Forgetting events classification Training")
    parser.add_argument('--architecture', type=str, default='resnet_18',
                        choices=['resnet_18', 'resnet_34', 'resnet_50', 'resnet_101'],
                        help='architecture name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='Kermany',
                        choices=['CIFAR10', 'Kermany'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--data_path', type=str,
                        default='/home/kaushik/Dropbox (GhassanGT)/OCT/BIGandDATA/ZhangData/OCT/',
                        help='dataset path')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # output directory for learning maps
    parser.add_argument('--out_dir', type=str, default='output/',
                        help='path to output directory for output files (excel, learning maps)')
    parser.add_argument('--recording_epoch', type=int, default=40,
                        help='evaluation interval (default: 1)')

    args = parser.parse_args()

    return args
