#allows viewing of a dataset compressed to sizze in config.ini

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os, os.path
import math
import configparser
from subprocess import call
import warnings

from anomaly_detection import label_target

warnings.simplefilter(action='ignore', category=FutureWarning)

config = configparser.ConfigParser()
config.read('config.ini')

dir_path = str(config.get("labelling", "directory"))
radius = int(config.get("labelling", "radius")) #search radius for assigning points to targets
size = int(config.get("labelling", "size"))
targetFlag = bool(config.get("view", "targets"))
pathFlag = bool(config.get("view", "lines"))

view_menu_options = {
    1: 'Select a dataset',
    2: 'Edit view options',
    3: 'Go back'
}

def print_menu():
    for key in view_menu_options.keys():
        print(key, '--', view_menu_options[key])

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

def get_data():
    file_count = count_files(dir_path)
    df1 = pd.DataFrame()
    while True:
        try:
            data = int(input("Choose data between 1 and {file_count}: ".format(file_count = file_count)))
            if 0 < data <= file_count:
                df = pd.read_csv("input/inputdata ({data}).csv".format(data = data), encoding ='unicode_escape')
                df.columns = ['X', 'Y']
                interval = int(1 / (size / len(df.index)))
                for i, row in df.iterrows():
                    if (i % interval) == 0:
                        df1 = df1.append(row)
                df1 = df1.reset_index(drop=True)
                drop = max(df1.index) % size
                df1 = df1[:-drop]
                return df1, data
        except (ValueError, TypeError):
            pass

def close_figure(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)

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
        least_x = 0
        lx = 0
        least_y = 0
        ly = 0
        for j in range(len(path[0])):
            if j != 0:
                if dist < least_distance:
                    least_distance = dist
                    least_x = lx
                    least_y = ly
            else:
                #least_distance is set to first computation for each new point
                least_distance = math.sqrt((path[0][0] + row['X'])**2 + (path[1][0] - row['Y'])**2)
            #dist = euclidean distance between path xy and row['X'], row['Y']
            dist = math.sqrt((path[0][j] - row['X'])**2 + (path[1][j] - row['Y'])**2)

            lx = path[0][j]
            ly = path[1][j]

        if (i % 1000 == 0):
            print("i: {i}, ld: {l}".format(i=i, l=least_distance))
            print(row['Target'])
            plt.plot([least_x, row['X']],[least_y,row['Y']], '-ro', label='min. Data -> Path')
        dx[i] = least_distance
    return dx

def create_image():
    df, data_index = get_data()
    title = ("Sample {i}, Size {s}".format(i = data_index, s = size))
    #title = ("Target Zones, Perfect Paths")
    plt.figure(figsize = (8,8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    #Plot movement data
    plt.scatter(x='X', y='Y', data=df, s=6)
    targets = build_targets()
    df['Target'] = label_target(df, targets)
    print(df.head())
    dx = avg_distance(df, targets)


    #Check if targets are to be plotted
    targetFlag = bool(config.get("view", "targets"))
    if targetFlag:
        print("Building targets...")
        for i in targets:
            plt.plot(i.x, i.y, marker="o", markersize="4", markeredgecolor="black", markerfacecolor="green")
            plt.plot(i.zoneBorder[0], i.zoneBorder[1], linestyle="solid", color="green")

    #Check if ideal path lines are to be plotted
    pathFlag = bool(config.get("view", "lines"))
    if pathFlag:
        targets = build_targets()
        print("Building lines...")
        pathLines = build_target_line(targets)
        for i in range(5):
            plt.plot(pathLines[i][0], pathLines[i][1], c='g')

    plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
    plt.show()

def edit_view_settings():
    call(["python", "config_files.py"])

def main():
    print("\n---VIEW DATASET---")
    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('\nEnter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
            create_image()
        elif option == 2:
            edit_view_settings()
        elif option == 3:
            print('Returning to main menu...\n')
            return
        else:
            print('\nInvalid option. Please enter a number between 1 and ', len(view_menu_options), "\n")
    create_image()

if __name__ == "__main__":
    main()
