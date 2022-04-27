import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from sklearn.preprocessing import KBinsDiscretizer
import configparser
import os, os.path
import math

config = configparser.ConfigParser()
config.read('config.ini')

dir_path = str(config.get("labelling", "directory"))

def count_files(path):
    count = 0
    for path in os.listdir(path):
        count += 1
    return count

def get_dataset(d, size):
    #returns a 1d array of format ([x1, y1, x2, y2, ..., xn, yn]) for dataset of size n
    file_count = count_files(dir_path)

    data = np.genfromtxt("input/inputdata ({i}).csv".format(i = d), delimiter = ",")
    interval = int(1 / (size/len(data)))
    data_trimmed = list()
    #print("Size: ", size)
    for i, row in enumerate(data):
        if (i % interval) == 0:
            data_trimmed.append(row)

    data_trimmed = np.asarray(data_trimmed)
    drop = int(len(data_trimmed) % size)
    data_trimmed = data_trimmed[:-drop]
    data_trimmed = data_trimmed.reshape(2*len(data_trimmed))
    return data_trimmed


def get_data_dxy(f):
    array = np.genfromtxt(f, delimiter=",")
    return array

def separate_csv(d):
    size = d[0]
    radius = d[1]
    dxy = d[2:]
    av_dxy = round(np.mean(dxy),2)
    return size, radius, dxy, av_dxy

def label_dex(dxy):
    mirror = [10,9,8,7,6,5,4,3,2,1]
    dex = np.zeros(len(dxy))
    dxy_r = dxy.reshape((-1,1))
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    Xt = discretizer.fit_transform(dxy_r)
    for i, x in enumerate(Xt):
        dex[i] =  mirror[int(x)]
    return dex

def create_dex_csv(dataset, dex, size, radius):
    #Creates a csv file of format:
    #Dataset Index, Parsed Size, Target Radius Used, Dexterity Classification, [x1,y1,x2,y2,...,xn,yn] where n=size
    file_count = count_files(dir_path)

    with open("dex_output/dataset_labelled-{s}-{r}.csv".format(s=int(size), r=int(radius)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(create_header(size))
        for i in range(1, file_count + 1):
            #print("Sample: {i}, Dex: {dex}".format(i=i, dex=dex[i-1]))
            data = get_dataset(i, size)
            #print(data)
            size_label = str(int(size))
            index_label = str(int(i))
            radius_label = str(int(radius))
            dexterity_label = str(int(dex[i-1]))

            title = [index_label, size_label, radius_label , dexterity_label]
            labelled_line = np.concatenate((title,data))
            print(labelled_line)
            writer.writerow(labelled_line)

def create_header(size):
    #size is multiplied by 2 since each point is represented by x and y
    size = size * 2
    header = list()
    header.append("index")
    header.append("size")
    header.append("radius")
    header.append("dexterity")
    for i in range(int(size)):
        if (i % 2 == 0): #even index therefore an x coord
            h = "point_{x}_x".format(x = int(((i/2)+1)))
            header.append(h)
        else:
            h = "point_{x}_y".format(x = int(math.ceil(i/2)))
            header.append(h)
    return np.asarray(header)

def choose_dataset():
    while True:
        try:
            dataset = input("Enter file name within directory /deviations/: ")
            dataset = ("deviations/{}".format(dataset))
            #print(dataset)
            if os.path.isfile(dataset):
                return dataset
            else:
                print("File doesn't exist")
        except (ValueError, TypeError):
            pass

def main():
    dataset = choose_dataset()
    data = get_data_dxy(dataset)
    #print(data)
    size, radius, dxy, av_dxy = separate_csv(data)
    print("Size: ", size)
    print("Radius: ", radius)
    print("Average Deviation: ", av_dxy)

    dex = label_dex(dxy)
    print(dex)
    plt.hist(dex)
    plt.title("Distribution of Dexterity Values")
    plt.show()
    #create_dex_csv(dataset, dex, size, radius)

if __name__ == "__main__":
    main()
