#creates a csv file using config.ini paramters
#to be used for generation of a pytorch DATASET

#SampleName, SampleSize, Dexterity, point1_x, point1,y,...,point[size]_x, point[size]_y

#this script will need to:
#1. Import all datasets sequentially
#2. Label each dataset with the "average devitation" value
#3. Find the average of averages for deviation
#4. Returns a list of all data samples below a threshold of average devitation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os, os.path
import math
import configparser
from subprocess import call
import csv

config = configparser.ConfigParser()
config.read('config.ini')

dir_path = str(config.get("labelling", "directory"))
radius = int(config.get("labelling", "radius")) #search radius for assigning points to targets
size = int(config.get("labelling", "size"))
targetFlag = bool(config.get("view", "targets"))
pathFlag = bool(config.get("view", "lines"))
lower_threshold = int(config.get("anomalies","lower"))
upper_threshold = int(config.get("anomalies", "upper"))

class Target:
    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.index = i #index of target (1,2,3,4,5)
        self.zoneBorder = self.constructBorder() #array of points describing circular zone around target point

    def __repr__(self):
        return ('\nTarget: {index}: X = {x}, Y = {y}'.format(index = self.index, x = self.x, y = self.y))

    def constructBorder(self):
        max_theta = 2 * np.pi
        list_t = list(np.arange(0, max_theta, 0.001))
        x_circle = [self.x + (radius * math.cos(x_y)) for x_y in list_t]
        y_circle = [self.y + (radius * math.sin(x_y)) for x_y in list_t]
        circle = np.stack((x_circle, y_circle))
        return circle

def build_targets():
#Adapted from HCAAR source code
    order = [1,3,5,2,4]
    sh = 1280
    sw = 1024
    a = int(sh / 2)
    b = int(sw / 2)
    r = 200
    t = np.zeros((5,2))
    for i in range(5):
        theta = (-90+(i*72)) * (math.pi / 180)
        x = a + (r * (math.cos(theta)))
        y = b + (r * (math.sin(theta)))
        t[i][0] = int(x)
        t[i][1] = int(y)

    targets = list()
    for i in range(5):
        targets.append(Target(t[i][0],t[i][1],order[i]))

    targets.sort(key=lambda x: x.index)
    return(targets)

def build_target_line(t):
    lines = list()
    fx = lambda x : (m*x + c)
    for i in range(5):
        lines.append(np.zeros((2, 100)))
        if i != 4:
            m = (t[i+1].y - t[i].y)/(t[i+1].x - t[i].x)
            c = (t[i].y - (m*t[i].x))
            lines[i][0] = np.linspace(t[i].x, t[i+1].x, 100)
            lines[i][1] = fx(lines[i][0])
        else:
            m = (t[0].y - t[i].y)/(t[0].x - t[i].x)
            c = (t[i].y - (m*t[i].x))
            lines[i][0] = np.linspace(t[i].x, t[0].x, 100)
            lines[i][1] = fx(lines[i][0])

    return (lines)

def count_files(path):
    count = 0
    for path in os.listdir(path):
        count += 1
    return count

def label_target(df, targets):
    #label each point with the current target
    #(If the last point to be struck was 1, each point is labelled
    #2 until 2 is struck)
    tv = np.zeros(len(df))
    for i, row in df.iterrows():
        if i == 0:
            tv[i] = 2
            currentTarget = 2
            continue
        if (((row['X'] - targets[currentTarget - 1].x) ** 2) + ((row['Y'] - targets[currentTarget - 1].y) ** 2)) < (radius ** 2):
            #hit
            if currentTarget == 5:
                currentTarget = 1
            else:
                currentTarget += 1
            tv[i] = currentTarget
        else:
            #still moving towards target
            tv[i] = currentTarget
    return tv

def avg_distance(df, t):
    #for each point, calculate the distance between
    #point and closest point on the regression line
    #between last target and next target
    lines = build_target_line(t) #list of numpy arrays containing points describing regression line between target and last target
    dx = np.zeros(len(df)) #distance between each point and "best path"
    for i, row in df.iterrows():
        for c, tx in enumerate(t):
            if tx.index == row['Target']: #get current target as target
                target = tx
                break

        path = lines[target.index - 1]
        least_distance = 0
        dist = 0
        for j in range(len(path[0])):
            if j != 0:
                if dist < least_distance:
                    least_distance = dist
            else:
                #least_distance is set to first computation for each new point
                least_distance = math.sqrt((path[0][0] + row['X'])**2 + (path[1][0] - row['Y'])**2)
            #dist = euclidean distance between path xy and row['X'], row['Y']
            dist = math.sqrt((path[0][j] - row['X'])**2 + (path[1][j] - row['Y'])**2)

        dx[i] = least_distance
    avg_dx = np.average(dx)
    return avg_dx

#Returns requested dataset of index i, sampled to size in config.ini
def get_data(d):
    file_count = count_files(dir_path)
    df1 = pd.DataFrame()
    while True:
        if (d > file_count):
            return("Error: Requested file is outside of range")
            pass
        try:
            df = pd.read_csv("input/inputdata ({data}).csv".format(data = d), encoding = 'unicode_escape')
            df.columns = ['X', 'Y']
            interval = int(1 / (size / len(df.index)))
            for i, row in df.iterrows():
                if (i % interval) == 0:
                    df1 = df1.append(row)
            df1 = df1.reset_index(drop=True)
            drop = max(df1.index) % size
            df1 = df1[:-drop]
            return df1
        except (ValueError, TypeError):
            pass

def get_data_size(index):
    file_count = count_files(dir_path)
    df = pd.DataFrame()
    while True:
        if (index > file_count):
            return("Error: Requested file is outside of range")
            pass
        try:
            df = pd.read_csv("input/inputdata ({data}).csv".format(data = index), encoding = 'unicode_escape')
            df.columns = ['X','Y']
            return df
        except (ValueError, TypeError):
            pass

def anomaly_check():
    file_count = count_files(dir_path)
    targets = build_targets()
    avg_dxy = np.zeros(file_count)
    anomalies = list()

    for i in range(1, file_count+1):
        df = get_data(i)
        if(len(df) < size):
            print("Set {i} omitted".format(i = i))
            continue
        df['Target'] = label_target(df, targets)
        avg_dxy[i-1] = round(avg_distance(df, targets), 2)
        if ((avg_dxy[i-1] < lower_threshold) or (avg_dxy[i-1] > upper_threshold)):
            anomalies.append([i, avg_dxy[i-1]])
        print("Dataset {i} av_dxy: {dxy}".format(i=i, dxy=avg_dxy[i-1]))

    print("Results:")
    print("Size: ", size, ", Radius: ", radius, ", Average deviation: ", np.average(avg_dxy),"\n")
    print("Anomalies:\n")
    for i in anomalies:
        print("I: {i}, AVG_DXY: {d}".format(i=i[0], d=i[1]))

    plt.hist(avg_dxy, bins = 10)
    plt.show()
    return avg_dxy

def create_avdxy_csv(avg_dxy):
    avg_dxy_list = list(avg_dxy)
    head = list([size, radius])
    content = head + avg_dxy_list
    #write to a new csv
    with open("deviations/deviations-{s}-{r}.csv".format(s=size, r = radius), 'w', encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(content)

def main():
    avg_dxy = anomaly_check()
    #create_avdxy_csv(avg_dxy)

if __name__ == "__main__":
    main()
