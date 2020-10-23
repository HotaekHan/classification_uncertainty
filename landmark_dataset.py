import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch.utils.data as data


class Landmark_dataset(data.Dataset):
    def __init__(self, root, is_train):
        if is_train is True:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'test')
        class_info_path = os.path.join(root, 'category.csv')

        category_df = pd.read_csv(class_info_path)
        self.class_dict = dict(category_df.values[:, ::-1])
        self.num_classes = len(self.class_dict)
        self.img_path = self._read_img_path(data_path)

        print("Number of classes : %d" % len(self.class_dict))
        print("Number of images : %d" % len(self.img_path))

    def _read_img_path(self, root_path):
        file_list = list()
        for (path, _, files) in os.walk(root_path):
            for file in files:
                ext = os.path.splitext(file)[-1]

                if ext == '.JPG':
                    file_list.append(os.path.join(path, file))

        return file_list

    def __getitem__(self, index):
        # img = cv2.imread(self.img_path[index]).astype(np.float32)
        img = cv2.imread(self.img_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.
        target_name = self.img_path[index].split(os.path.sep)[-2]
        target = self.class_dict[target_name]

        # img = img.transpose((1, 2, 0))
        img = Image.fromarray(img)

        return img, target

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])

    # transform_train = transforms.Compose([
    #     transforms.Resize(size=(256, 256)),
    #     transforms.RandomCrop(size=(256, 256), padding=4),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = Landmark_dataset(root='/data/kaggle/dacon_landmark_korea/public', transform=transform_train)



    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=True)

    for imgs, targets in data_loader:
        tmp=0