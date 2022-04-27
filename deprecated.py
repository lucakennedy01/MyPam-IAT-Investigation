import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import os, os.path
import math

#old shortest distance to target path function
#Reason for removal:
#Distance check looks for points beyond the scope
#of the last and next target points, meaning distances
#can be calculated looking for points that are not
#on the path.
#Also throw NaN's for points exactly parallel (placed on)
#the path (division by zero), which should instead return a distance of zero

#Takes dataframe (df) and list of target objects (t)
def describe_dexterity(df, t):
    #for each point, calculate the distance between
    #point and closest point on the regression line
    #between last target and next target
    para_d = np.zeros(len(df)) #distance between each point and "best path"
    for i, row in df.iterrows():
        #check euclidean distance between rowx, rowy and closest point
        #on row 'Target' and 'Target' - 1
        intersect = np.zeros(2)
        #create function describing line between t-1 and #t
        #t[0] -> target.index = 1
        #t[1] -> target.index = 2
        #t[2] -> target.index = 3
        #t[3] -> target.index = 4
        #t[4] -> target.index = 5
        for c, tx in enumerate(t):
            if tx.index == row['Target']: #get current target as target
                target = tx
                if tx.index == 1:
                    lastTarget = t[4]
                else:
                    lastTarget = t[c-1]
                break

        m = (target.y - lastTarget.y)/(target.x - lastTarget.x) #gradient of line between last target and next target
        m_p = -(1 / m) #gradient of line perpendicular to target line (inverse of m)
        c_t = (target.y - (m*target.x))
        c_p = (row['Y'] - (m_p*row['X']))
        fxt = lambda x : (m*x + c_t)
        #fxp = lambda x : (m_p*x + c_p)
        #find point of intersection
        intersect[0] = (c_p - c_t)/(m - m_p)
        intersect[1] = fxt(intersect[0])
        #calculate euclidean distance between row['X'], row['Y'] and intersect
        d_x = math.sqrt((intersect[0] - row['X'])**2 + (intersect[1] - row['Y'])**2)
        para_d[i] = (d_x)

    return para_d

#Used to trim data to one minute
#Reason for removal:
#In early stages I was manually adding time
#to csv data before importing. Due to inconsistencies in data,
#sometimes values exceeding 60000ms would be assigned to point data
#This function was used to remove data past 60000ms and return
#the "trimmed" dataframe
def trim_to_minute(df):
    #returns only 60 seconds worth of information
    total_rows = len(df.index)
    max_time = df['Time'].iloc[-1]
    final_index = int((60 / max_time) * total_rows)
    final_index_trim = (total_rows - final_index)
    df = df[:-final_index_trim]
    return df
