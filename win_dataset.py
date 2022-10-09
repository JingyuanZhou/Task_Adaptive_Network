import random
import numpy as np
import glob
from PIL import Image
import math
from typing import Callable

import torch
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset
import torchvision
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torch.utils.data import  RandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_rain(orig_images_path, degen_images_path, mode, repeats=1):
    ls = []
    image_list_degen = glob.glob(degen_images_path + mode +"*.jpg")
    image_list_degen = [i.replace('\\','/') for i in image_list_degen]
    for image_degen_path in image_list_degen:
        image = image_degen_path.split("/")[-1]
        key = image.split("_")[0]+'.jpg'
        image_clear_path=orig_images_path + mode + key
        for _ in range(repeats):
            ls.append([image_clear_path,image_degen_path,0])
    return ls

def get_hazy(orig_images_path, degen_images_path, mode, repeats=1):
    ls = []
    image_list_degen = glob.glob(degen_images_path + mode +"*.png")
    image_list_degen = [i.replace('\\','/') for i in image_list_degen]
    for image_degen_path in image_list_degen:
        image = image_degen_path.split("/")[-1]
        key = image.split("_")[0]+'.png'
        image_clear_path=orig_images_path + mode + key
        for _ in range(repeats):
            ls.append([image_clear_path,image_degen_path,1])
    return ls

def get_rain_drop(orig_images_path, degen_images_path, mode, repeats=1):
    ls = []
    postfix = 'jpg' if mode == 'rain_drop_test_b/' else 'png'
    image_list_degen = glob.glob(degen_images_path + mode +f"*.{postfix}")
    image_list_degen = [i.replace('\\','/') for i in image_list_degen]
    for image_degen_path in image_list_degen:
        image=image_degen_path.split("/")[-1].split("_")[0]
        image_clear_path=orig_images_path + mode + image+f"_clean.{postfix}"
        for _ in range(repeats):
            ls.append([image_clear_path,image_degen_path,2])
            
    return ls

def get_low_light(orig_images_path, degen_images_path, mode, repeats=1):
    ls = []
    image_list_degen = glob.glob(degen_images_path + mode +"*.png")
    image_list_clear = glob.glob(orig_images_path + mode +"*.png")
    image_list_degen = [i.replace('\\','/') for i in image_list_degen]
    image_list_clear = [i.replace('\\','/') for i in image_list_clear]
    for image_degen_path in image_list_degen:
        image=image_degen_path.split("/")[-1].split("_")[0]
        if orig_images_path + mode + image+"_00_30s.png" in image_list_clear:
            image_clear_path = orig_images_path + mode + image+"_00_30s.png"
        else:
            image_clear_path = orig_images_path + mode + image+"_00_10s.png"
        for _ in range(repeats):
            ls.append([image_clear_path,image_degen_path,3])
    
    return ls

def get_blur(orig_images_path, degen_images_path, mode, repeats=1):
    ls = []
    image_list_degen = glob.glob(degen_images_path + mode +"*.png")
    image_list_degen = [i.replace('\\','/') for i in image_list_degen]
    for image_degen_path in image_list_degen:
        image=image_degen_path.split("/")[-1]
        image_clear_path=orig_images_path + mode +image
        for _ in range(repeats):
            ls.append([image_clear_path,image_degen_path,4])
            
    return ls


def populate_train_list(orig_images_path, degen_images_path):
    all_list=[]

    sub_datasets=['rain/','hazy/','rain_drop/']

    for sub in sub_datasets:
        if sub == 'rain/':
            tmp_ls = get_rain(orig_images_path, degen_images_path, sub)
        
        elif sub == 'hazy/':
            tmp_ls = get_hazy(orig_images_path, degen_images_path, sub)
        
        elif sub == 'rain_drop/':
            tmp_ls = get_rain_drop(orig_images_path, degen_images_path, sub, 12)
        '''
        elif sub == 'low_light/':
            tmp_ls = get_low_light(orig_images_path, degen_images_path, sub, 4)
                
        elif sub == 'blur/':
            tmp_ls = get_blur(orig_images_path, degen_images_path, sub, 3)
        '''
        all_list.extend(tmp_ls)
    return all_list

def populate_test_list(orig_images_path, degen_images_path,mode):
    all_list = []
    mode=mode+'/'
    if mode == 'rain/':
        all_list = get_rain(orig_images_path, degen_images_path,mode)
        
    elif mode == 'hazy/':
        all_list = get_hazy(orig_images_path, degen_images_path,mode)
    
    elif mode == 'low_light/':
        all_list = get_low_light(orig_images_path, degen_images_path,mode)
    
    elif mode == 'blur/':
        all_list = get_blur(orig_images_path, degen_images_path,mode)
                
    elif mode == 'rain_drop/' or mode == 'rain_drop_test_a/' or mode == 'rain_drop_test_b/':
        all_list = get_rain_drop(orig_images_path, degen_images_path,mode)
    
    return all_list

class merge_dataset(data.Dataset):
    def __init__(self,data_list, train=True):
        self.data_list = data_list
        self.train = train
        title = 'training' if self.train  else 'testing'
        print("Total "+title+" examples:", len(self.data_list))
        self.outsize = (64,64) if self.train else (256,256)
        self.train_transform = A.Compose([
            A.Resize(256, 256),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.test_transform = A.Compose([
            A.Resize(256, 256),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.transform = self.train_transform if self.train else self.test_transform

    def __getitem__(self, index):

        data_orig_path, data_degen_path,label = self.data_list[index]

        data_orig = cv2.imread(data_orig_path)
        data_orig = cv2.cvtColor(data_orig, cv2.COLOR_BGR2RGB)
        data_orig = data_orig / 255.0
        data_orig = self.transform(image=data_orig)["image"]

        data_degen = cv2.imread(data_degen_path)
        data_degen = cv2.cvtColor(data_degen, cv2.COLOR_BGR2RGB)
        data_degen = data_degen / 255.0
        data_degen = self.transform(image=data_degen)["image"]

        return data_orig.float(), data_degen.float(), label

    def _get_label(self, index):
        _, _,label = self.data_list[index]
        return label
    
    def __len__(self):
        return len(self.data_list)
    
  
# ----------------------------------------------------------------------------------------------------------------------
  
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class ExampleImbalancedDatasetSampler(ImbalancedDatasetSampler):
    """
    ImbalancedDatasetSampler is taken from:
    https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    In order to be able to show the usage of ImbalancedDatasetSampler in this example I am editing the _get_label
    to fit my datasets
    """
    def _get_label(self, dataset, idx):
        return dataset._get_label(idx)

class BalancedBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a balanced batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if dataset_idx == 0:
                # the first dataset is kept at RandomSampler
                sampler = RandomSampler(cur_dataset)
            else:
                # the second unbalanced dataset is changed
                sampler = ExampleImbalancedDatasetSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
    
# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    '''
    config = {}
    config['orig_test_images_path']='/home/mist/zs/data/merge_test/clear/'
    config['degen_test_images_path']='/home/mist/zs/data/merge_test/degen/'
    rain_list = populate_test_list(config['orig_test_images_path'],config['degen_test_images_path'],'rain')
    hazy_list = populate_test_list(config['orig_test_images_path'],config['degen_test_images_path'],'hazy')

    first_dataset = merge_dataset(rain_list,train=False)
    second_dataset = merge_dataset(hazy_list,train=False)
    concat_dataset = ConcatDataset([first_dataset, second_dataset])

    batch_size = 4

    # dataloader with BalancedBatchSchedulerSampler
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                            sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
                                                                                  batch_size=batch_size),
                                            batch_size=batch_size,
                                            shuffle=False)
    
    for i,inputs in enumerate(dataloader):
        print(inputs[2])
        if i>5:
            break
    '''