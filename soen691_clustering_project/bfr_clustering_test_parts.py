import glob
import os
from pathlib import Path
from sklearn.datasets import make_blobs
import time
import numpy as np 
import math
import matplotlib.pyplot as plt 

from bfr import BFR


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
    data_gen, y = make_blobs(n_samples=2500, n_features=20, centers=nb_cts, cluster_std=r_stds, center_box=(-10,10))

    print("Data Size: ", data_gen.shape)
    #estimate filesize
    fsize = data_gen.size * data_gen.itemsize
    #save to file
    np.savetxt(fname, data_gen, delimiter=delim)

    #clustering

    #creating partitions
    splits = np.arange(1, 10)
    times = np.zeros(splits.size)

    for i in range(times.size):

        print("Clustering, chunk size: ", fsize / splits[i], " aka ", splits[i])

        #averaging results over several iterations

        itr = 10

        for j in range(itr):
            print("Clustering, chunk size: ", fsize / splits[i], " aka ", splits[i], " iteration: ", j)

            bfr_c1 = BFR()

            st = time.time()

            bfr_c1.cluster_partition(fname, fsize / splits[i], delim, nb_cts, 1.5, 1.5)

            et = time.time()
            #add to times
            times[i] += (et - st)

            print("Complete. Time: ", et - st)

        #average
        times[i] = times[i] / itr

    print("List of Times")
    print(times)

    #plotting

    plt.plot(splits, times, 'r-')
    plt.axis([0,10,0,60])
    plt.xlabel("Number of Partitions")
    plt.ylabel("Execution Time (s)")
    plt.title("BFR clustering with partitioned data")
    plt.show()

