import os
import sys
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register
import numpy as np

@register('mini-imagenet')
class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        self.split_tag = split
        if split not in ['train', 'val', 'test']:
            raise NotImplementedError
        print("Okay, ", self.split_tag)


        data_path = os.getcwd() + "/materials/mini-imagenet/" + split

        print("Data path: ", data_path)
        print("Contents: ", os.listdir(data_path))
        data = []
        label = []
        for count, class_id in enumerate(os.listdir(data_path)):
            class_label = count
            class_path = os.path.join(data_path, class_id)
            for class_image in os.listdir(class_path):
                clas_image_path = os.path.join(class_path, class_image)
                class_image_label = class_label
                data.append(clas_image_path)
                label.append(class_image_label)
            print("Class: ", class_label, class_id, " Done")
                # print("Class Image label: ", class_image_content.shape)
            # print("class content: ", os.listdir(class_path))
            # sys.exit()




        # Need to customize from here

        # split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        # with open(os.path.join(root_path, split_file), 'rb') as f:
        #     pack = pickle.load(f, encoding='latin1')
        # data = pack['data']
        # label = pack['labels']

        image_size = 80

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        #No more need of augment resize
        print(f"{split} data prep done!")

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean

        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def get_data(self, i):
        image_path = self.data[i]
        image = np.array(Image.open(image_path))
        return Image.fromarray(image)

    def __getitem__(self, i):
        the_data = self.get_data(i)
        # print("self.transform(self.data[i]): ", self.transform(the_data).shape)
        return self.transform(the_data), self.label[i]


class MiniImageNet_old_for_reference(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']

        image_size = 80
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean

        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]
