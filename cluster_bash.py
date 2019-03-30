'''
This is a bash.py script to get used to the new MeterDataSource format of
extracting meter data from EL
'''

# from utils.downloader import MeterDataSource
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
import pandas as pd
import re
import math

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random

def get_data(filepath):
    return pd.read_csv(filepath, index_col='Timestamps',
                        usecols=['Value', 'Timestamps'])

def convert_time(df):

    timestamps = df.index.tolist()
    time = []

    for item in timestamps:
        raw_time = re.split('[T +]', item)[1]
        if raw_time[3] == '3':
            time.append(int(raw_time[:2]) + 0.5)
        else:
            time.append(int(raw_time[:2]))

    time_x = []
    time_y = []
    for t in time:
        time_x.append(math.sin((t*15)*(math.pi/180)))
        time_y.append(math.cos((t*15)*(math.pi/180)))

    df['time_x'] = time_x
    df['time_y'] = time_y

    return df

def normalise_values(df):

    values = df['Value'].tolist()
    norm = []
    amin, amax = min(values), max(values)
    for i, val in enumerate(values):
        norm.append((val-amin) / (amax-amin))

    df['norm_vals'] = norm
    # df['norm_vals'] = values # Sometimes it seems better to use the actual real power values
    # print norm
    return df

def plot_stuff(df):
    x = df['time_x'].tolist()
    y = df['time_y'].tolist()
    z = df['norm_vals'].tolist()

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    pyplot.show()

def new_plot_stuff(df):
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    z0 = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []

    for index, row in df.iterrows():
        if row['cluster_num'] == 0:
            x0.append(row['time_x'])
            y0.append(row['time_y'])
            z0.append(row['norm_vals'])
        elif row['cluster_num'] == 1:
            x1.append(row['time_x'])
            y1.append(row['time_y'])
            z1.append(row['norm_vals'])
        elif row['cluster_num'] == 2:
            x2.append(row['time_x'])
            y2.append(row['time_y'])
            z2.append(row['norm_vals'])
        elif row['cluster_num'] == 3:
            x3.append(row['time_x'])
            y3.append(row['time_y'])
            z3.append(row['norm_vals'])
        elif row['cluster_num'] == 4:
            x4.append(row['time_x'])
            y4.append(row['time_y'])
            z4.append(row['norm_vals'])

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(x0, y0, z0, c='g')
    ax.scatter(x1, y1, z1, c='b')
    ax.scatter(x2, y2, z2, c='r')
    ax.scatter(x3, y3, z3, c='y')
    ax.scatter(x4, y4, z4, c='m')
    pyplot.show()

def main():

    raw_pandas = get_data('test_rhac.csv')
    # print raw_pandas
    print 'raw pandas collected'

    circle_time_pandas = convert_time(raw_pandas)
    # print circle_time_pandas
    print 'circle_time_pandas converted'
    # circle_time_pandas.plot(x='time_x', y='time_y', style='o')
    # plt.show()

    norm_pandas = normalise_values(circle_time_pandas)
    norm_pandas.drop('Value', 1, inplace=True)
    print 'values are normalised'

    ''' Affinity Propogation Clustiner'''
    affprop = AffinityPropagation().fit(norm_pandas.values)
    labels = affprop.labels_
    print labels

    ''' DBScan Clustering'''
    # dbscan = DBSCAN(eps=0.4).fit(norm_pandas.values)
    # labels = dbscan.labels_
    # print len(labels)
    # print labels

    ''' Kmeans Clustering'''
    # kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(norm_pandas.values)
    # labels = kmeans.labels_
    # inertia = kmeans.inertia_
    # print labels
    # print inertia

    norm_pandas['cluster_num'] = labels
    # print norm_pandas

    new_plot_stuff(norm_pandas)

    # clust_0 = []
    # clust_1 = []
    # for i in range(len(norm_pandas.values)):
    #     # print norm_pandas.values[i], labels[i]
    #     if labels[i] == 0:
    #         clust_0.append(norm_pandas.values[i])
    #     elif labels[i] == 1:
    #         clust_1.append(norm_pandas.values[i])
    #
    # print clust_0
    # print clust_1

    # plot_stuff(norm_pandas)

main()
