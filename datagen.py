import random
import numpy as np
import json
import os
import cv2
import albumentations as A
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class jsonDataset(data.Dataset):
    def __init__(self, path, classes):
        self.path = path
        self.classes = classes

        self.fnames = list()
        self.labels = list()

        self.num_classes = len(self.classes)
        self.label_map = dict()
        for class_idx, class_name in enumerate(self.classes):
            self.label_map[class_name] = class_idx

        fp_read = open(self.path, 'r')
        gt_dict = json.load(fp_read)

        all_labels = list()
        all_img_path = list()

        # read gt files
        for gt_key in gt_dict:
            gt_data = gt_dict[gt_key][0]

            # img = cv2.imread(gt_data['image_path'])
            # img_rows = img.shape[0]
            # img_cols = img.shape[1]

            class_name = gt_data['label']
            if class_name not in self.classes:
                print('weired class name: ' + class_name)
                print(gt_data['image_path'])
                continue

            class_idx = self.label_map[class_name]
            all_labels.append(class_idx)
            all_img_path.append(gt_data['image_path'])

        if len(all_labels) == len(all_img_path):
            num_images = len(all_img_path)
        else:
            print('num. of labels: ' + str(len(all_labels)))
            print('num. of paths: ' + str(len(all_img_path)))
            raise ValueError('num. of elements are different(all boxes, all_labels, all_img_path)')

        for idx in range(0, num_images, 1):
            self.fnames.append(all_img_path[idx])
            self.labels.append(torch.tensor(all_labels[idx], dtype=torch.int64))

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        labels = self.labels[idx]
        # img = cv2.imread(fname)
        img = Image.open(fname)

        return img, labels, fname

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    # ])
    # set random seed
    random.seed(3000)
    np.random.seed(3000)
    torch.manual_seed(3000)
    img_size = (32, 32)

    transform_train = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.RandomCrop(size=img_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # classes = 'aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor'
    classes = 'car|bus|truck'
    classes = classes.split('|')

    dataset = jsonDataset(path='data/its_train_split.json', classes=classes)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)

    while True:
        for idx, (images, targets, paths) in enumerate(dataloader):
            np_img = images.numpy()
            print(images.size())
            print(targets.size())
            print(paths.size())
            # break

if __name__ == '__main__':
    test()
