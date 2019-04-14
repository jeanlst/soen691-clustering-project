import glob
import os
from pathlib import Path
from sklearn.datasets import make_blobs
import time
import numpy as np 
import math
import matplotlib.pyplot as plt 

from bfr import BFR
from kmeans import KMeans
from cure import Cure


if __name__ == "__main__":

    #generating data

    #location for data to save / load
    fname = Path("./data/highdim.csv")
    delim = ","

    #creating dataset

    #nb centers
    nb_cts = 5

    #randomized standard deviations between 1 and 3
    r_stds = (np.random.rand(nb_cts) * 0.8) + 0.7

    #1000 lines, 10 features, 3 centres
    data_gen, y = make_blobs(n_samples=3000, n_features=20, centers=nb_cts, cluster_std=r_stds, center_box=(-5,5))

    #print("Data Size: ", data_gen.shape)
    #estimate filesize
    fsize = data_gen.size * data_gen.itemsize
    #save to file
    np.savetxt(fname, data_gen, delimiter=delim)

    #clustering

    #creating partitions
    splits = np.arange(1, 10)
    times = np.zeros(splits.size)

    #TEST 1: Plotting Execution Time of BFR vs number of splits in the data

    for i in range(times.size):

        print("Clustering, chunk size: ", fsize / splits[i], " aka ", splits[i])

        #averaging results over several iterations

        itr = 10

        bfr_c1 = BFR()

        for j in range(itr):
            print("Clustering, chunk size: ", fsize / splits[i], " aka ", splits[i], " iteration: ", j)

            st = time.time()

            bfr_c1.cluster_partition(fname, fsize / splits[i], delim, nb_cts, 1.5, 1.5, False)

            et = time.time()
            #add to times
            times[i] += (et - st)

            print("Complete. Time: ", et - st)

        #average
        times[i] = times[i] / itr

    #print("List of Times")
    #print(times)

    #TEST 2: Plotting execution time of BFR at optimal split and comparing to K-Means

    #K-means plotting

    #averaging over 10 itrs

    km_avg = 0

    print("Clustering K-means")

    for i in range(10):

        print("Kmeans itr ", i)

        st = time.time()

        #load data into mem
        hd_data = np.loadtxt(fname, delimiter=delim)

        kmeans = KMeans(hd_data.tolist(), nb_cts)

        #cluster
        kmeans.clustering()

        et = time.time()

        km_avg += (et - st)

        #remove data from mem
        del hd_data

    km_avg = km_avg / 10

    #gathering data for bar charts
    bar_x = np.arange(2)
    c_times = [np.min(times), km_avg]

    #plotting

    plt.subplot(211)
    plt.plot(splits, times, 'r-')
    plt.plot(splits, times, 'ro')
    plt.axis([0,9,0,np.amax(times)])
    plt.xlabel("Number of Partitions")
    plt.ylabel("Execution Time (s)")
    plt.title("BFR clustering with partitioned data")

    plt.subplot(212)
    plt.bar(bar_x, c_times)
    plt.xticks(bar_x, ('BFR', 'K-Means'))
    plt.xlabel("Type of Clustering Method")
    plt.ylabel("Execution Time (s)")
    plt.title("Comparison of Clustering methods on large dataset")
    plt.show()

