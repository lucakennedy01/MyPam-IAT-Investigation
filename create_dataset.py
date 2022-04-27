#Creates a pytorch compatible dataset representing
#all samples of data as follows:
#SampleIndex, Dexterity, point1_x, point1,y,...,point[size]_x, point[size]_y

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import warnings

__spec__ = None

def show_points(points):
    plt.scatter(points[:, 0], points[:, 1], s = 6)
    plt.pause(0.001)

class PentagramPointsDataset(Dataset):
    #Dataset representing points describing path of motion
    def __init__(self, csv_file, transform=None):
        self.pentagram_points = pd.read_csv(csv_file, encoding='unicode_escape')
        self.transform = transform

    def __len__(self):
        return len(self.pentagram_points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #each sample to be labelled with its dexterity
        sample_label = self.pentagram_points.iloc[idx, 3]
        sample_path_points = self.pentagram_points.iloc[idx, 4:]
        sample_path_points = np.array([sample_path_points])
        sample_path_points = sample_path_points.astype('float').reshape(-1,2)
        sample = [sample_path_points, sample_label]

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    #Convert ndarrays in sample to Tensors
    def __call__(self, sample):
        label, points = sample[1], sample[0].astype(np.float32)
        return [torch.from_numpy(points), label]

#plt.ion()   # interactive mode

def view_set():
    pentagram_dataset = PentagramPointsDataset(csv_file = "dex_output/dex_labelled-s1000-r30.csv")

    fig = plt.figure(figsize = (16,6))
    for i in range(len(pentagram_dataset)):
        sample = pentagram_dataset[i]

        #print(i, sample['label'], sample['points'].shape)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{i}, Dex: {d}'.format(i = i, d = sample['label']))
        ax.axis('off')
        show_points(sample['points'])

        if i == 2:
            plt.show()
            break

#Helper function to show a batch
def show_points_batch(sample_batched):
    dex_batch, points_batch = sample_batched['label'], sample_batched['points']
    batch_size = len(dex_batch)

    for i in range(batch_size):
        plt.scatter(points_batch[i, :, 0].numpy(), points_batch[i, :, 1].numpy(), s=6)
        plt.title('Batch from dataloader')


def transform_dataset(csv):
    transformed_dataset = PentagramPointsDataset(csv_file = csv, transform = ToTensor())

    #for i in range(len(transformed_dataset)):
    #    sample = transformed_dataset[i]
    #    print(i, sample['label'], sample['points'].size())
    return transformed_dataset

def get_labelled_file():
    while True:
        try:
            dataset = input("Enter file name within directory /dex_output/: ")
            dataset = ("dex_output/{}".format(dataset))
            #print(dataset)
            if os.path.isfile(dataset):
                return dataset
            else:
                print("File doesn't exist")
        except (ValueError, TypeError):
            pass

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    warnings.filterwarnings("ignore")

    #view_set()
    csv = get_labelled_file()
    transformed_dataset = transform_dataset(csv)

    #Map-Style Dataset
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    #for i_batch, sample_batched in enumerate(dataloader):
    #    print(i_batch, sample_batched['label'], sample_batched['points'])

        #observe 4th batch then stop
        #if i_batch == (4):
        #    plt.figure()
        #    show_points_batch(sample_batched)
        #    plt.axis('off')
        #    plt.show()

if __name__ == "__main__":
    main()

#samples = pd.read_csv("dex_output/dex_labelled-s1000-r30.csv", encoding ='unicode_escape')
#samples.index += 1 #dataframe index now matche callable sample index

#A sample's can be viewed through sample.iloc[dataframe-index]
#Sample = 66 #sample we want to view
#n = Sample - 1 #indexing starts at 0 for using iloc
#Get sample index
#sample_index = samples.iloc[n,0]
#Get size
#sample_size = samples.iloc[n,1]
#Get target radius used for generating dexterity
#sample_target_radius = samples.iloc[n,2]
#Get dexterity label
#sample_dex = samples.iloc[n,3]
#Get coordinates of points for a sample
#sample_plot = sample.iloc[n, 4:]
#sample_plot = np.asarray(sample_plot).reshape(-1,2) #can be plotted with x = sample_plot[:,0], y = sample_plot[:,1]
#print(data_points)
