import sys
import pickle
import os
import numpy as np
from PIL import Image

import torch.utils.data as data

class CIFAR_split(data.Dataset):
    def __init__(self, dir_path, num_exclude, train):
        self.data = list()
        self.targets = list()
        filelist = self.get_file_list(dir_path, train)

        for file_path in filelist:
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = np.array(self.targets)
        include_idx = self.targets < num_exclude

        self.targets = self.targets[include_idx]
        self.data = self.data[include_idx]

    def get_file_list(self, dir_name, is_train):
        filelist = list()

        is_cifar10 = False
        if 'cifar-10-' in dir_name:
            is_cifar10 = True

        if is_cifar10 is True:
            if is_train is True:
                for iter_idx in range(1, 6):
                    filelist.append(os.path.join(dir_name, 'data_batch_' + str(iter_idx)))
            else:
                filelist.append(os.path.join(dir_name, 'test_batch'))
        else:
            if is_train is True:
                filelist.append(os.path.join(dir_name, 'train'))
            else:
                filelist.append(os.path.join(dir_name, 'test'))

        return filelist

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data = CIFAR_split(dir_path='cifar-10-batches-py', num_exclude=50, train=True)

    tmp=0


