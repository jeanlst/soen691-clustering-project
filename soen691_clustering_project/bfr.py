
"""BFR implementation"""
import numpy as np
import math
from cluster import Cluster

class BFR:

    def __init__(self, data=None, k=None, alpha=1, beta=1):
        """
        DATA: numpy matrix
        K: Number of Clusters
        ALPHA: for Primary Compression. Number of standard deviations from the centroid under
        which points are grouped together and summarized
        BETA: for Secondary Compression. Merge groups of points where their "tightness" is under
        this value
        """

        self.__data = np.array(data) if isinstance(data, list) else data
        self.__k = k
        self.__alpha = alpha
        self.__beta = beta
        #Data Dimensionality
        self.__dims = None
        #list of Clusters
        self.__clusters = []

    def __convertStoG(self, mtx):
        '''
        Convert a matrix of singleton points into a matrix group of (SUM, SUMSQ, N)
        '''

        #get number of rows and columns
        rows = mtx.shape[0]
        cols = mtx.shape[1]

        #set number of cols and rows for new mtx
        nrows = rows
        ncols = cols * 2 + 1

        #new mtx
        nmtx = np.zeros((nrows, ncols))

        #set values
        nmtx[:,:cols] = mtx * 1.0
        nmtx[:,cols:-1] = mtx ** 2.0
        nmtx[:,-1] = 1 * 1.0

        return nmtx

    def __dist_mean(self, row1, row2, dims):
        '''
        distance calculation for grouped values
        '''

        #number of elements
        n1 = row1[dims * 2]
        n2 = row2[dims * 2]

        #means
        mean1 = (row1 * 1.0) / n1
        mean2 = (row2 * 1.0) / n2

        #distance calculation
        #squared euclidean

        total_dist = 0

        for i in list(range(dims)):
            total_dist += (mean1[i] - mean2[i]) ** 2.0

        return total_dist

    def __closest(self, row, centroids, dims):
        '''
        Given a row and a matrix of centroids, return index number
        for closest centroid
        dims: number of dimensions
        '''
        #assign distance to row
        #leave out ID list for centroids
        distances = np.apply_along_axis(lambda x: self.__dist_mean(row, x, dims), 1, centroids[:,:-1])

        #get index of closest
        min_index = np.argmin(distances)

        #return 'ID no.' of closest centroid
        return centroids[min_index,-1]

    def __cmean(self, row, dims):
        """
        given a row, return the mean
        """

        nrow = (row[:dims] * 1.0) / row[dims * 2]

        return nrow

    def __centroid_selection(self, mtx, k, dims):
        '''
        Select centroid at random, ensuring that the selected centroids are all 
        different values
        '''

        #attempt to find centroids that are unique (no duplicate rows)
        unique_cts = False

        itr = 0

        while unique_cts == False:

            #select k points
            indices = np.random.choice(mtx.shape[0], k, replace=False)

            #get the centroids
            selected = mtx[indices,:]

            #get means
            sel_means = np.apply_along_axis(lambda x: self.__cmean(x, dims), 1, selected)

            #should be more robust way to check uniqueness

            #filter for uniqueness
            unique_cents = np.unique(sel_means, axis=0)

            #check if unique and orig are same size
            if selected.shape[0] == unique_cents.shape[0]:
                unique_cts = True

            itr += 1

            if itr > 100:
                print("Unable to find unique centroids!")
                break

        return selected

    def __kmeans_groups_init(self, mtx, k, dims):
        '''
        perform weighed kmeans with groups of values
        choose initial centroids
        '''

        centroids = self.__centroid_selection(mtx, k, dims)
        #index numbers for the newly selected centroids
        c_inds = np.arange(k).reshape((k,1))
        #append indices to end of centroid list
        centroids = np.hstack((centroids, c_inds))

        #assign to each point closest centroid
        #column vector for assignments
        assignments = np.apply_along_axis(lambda x: self.__closest(x, centroids, dims), 1, mtx)

        #matrix plus assignments
        assigned = np.hstack((mtx, assignments.reshape(assignments.shape[0], 1)))

        return assigned, centroids

    def __recenter(self, mtx, dims):
        '''
        given a matrix of assigned points, average points and return new centroids
        '''

        #get all centroid IDs

        cent_IDs = np.unique(mtx[:,-1])

        #calculate averages

        current = np.zeros((cent_IDs.shape[0], mtx.shape[1]))

        #creating a dictionary to associate IDs with indices
        cent_dict = {k: cent_IDs[k] for k in list(range(len(cent_IDs)))}

        #for each unique value / centroid
        #dont see how to do this without a loop
        for i in list(range(len(cent_IDs))):

            #slicing for current value
            cind = np.where(mtx[:,-1] == cent_dict[i])[0]
            #selecting
            c_slice = mtx[cind,:]
            #sum
            c_sum = np.sum(c_slice, 0)
            #set to index
            current[int(i),:-1] = c_sum[:-1]
            #set last slot (index)
            current[int(i),-1] = cent_dict[i]

        return current

    def __centroid_comp(self, cts1, cts2, dims):
        '''
        compare 2 lists of centroids, 
        check if identical, regardless of number of
        members in the subgroup
        '''

        #check if centroids are the same
        same_cents = True

        for i in range(cts1.shape[0]):
            #get averages for cluster center
            cls_c1 = cts1[i,:dims] / cts1[i,dims*2]
            cls_c2 = cts2[i,:dims] / cts2[i,dims*2]
            #equality check
            if np.array_equal(cls_c1, cls_c2) == False:
                same_cents = False
        
        return same_cents

    def __matrix_insert(self, old_c, new_c):
        """
        Attempt to fix the "dissapearing centroid" issue.
        reinsert discarded centroid into new matrix
        """
        #get IDs for centroids
        old_ids = np.unique(old_c[:,-1])
        new_ids = np.unique(new_c[:,-1])

        #find missing indices
        missed = np.setdiff1d(old_ids, new_ids)

        #create empty matrix, same size as old
        reconst = np.zeros((old_c.shape[0], old_c.shape[1]))
        #fill it
        #new values
        for i in new_ids:
            #get relevant indexes
            rind = np.where(new_c[:,-1] == i)[0][0]
            r2ind = np.where(old_c[:,-1] == i)[0][0]
            reconst[r2ind,:] = new_c[rind,:]
        for i in missed:
            #get relevant index
            rind = np.where(old_c[:,-1] == i)[0][0]
            reconst[rind,:] = old_c[rind,:] 

        #return reconstructed mtx
        return reconst


    def __kmeans_converge(self, mtx, cents, k, dims):
        '''
        given a set of assigned points, average and reassign 
        until convergeance
        '''

        converge = False

        while converge == False:

            #get new centroids from average of 
            new_cents = self.__recenter(mtx, dims)

            if cents.shape[0] != new_cents.shape[0]:
                #print("disappearing centroid")
                #attempting to fix the "Disapearing Centroid" problem
                new_cents = self.__matrix_insert(cents, new_cents)
    
                
            if self.__centroid_comp(cents, new_cents, dims) == True:
                #centroids are equivalent, convergeance gained
                converge = True
            else:
                #reassign
                reassign = np.apply_along_axis(lambda x: self.__closest(x, new_cents, dims), 1, mtx[:,:-1])
                #orig matrix plus assignments
                mtx = np.hstack((mtx[:,:-1], reassign.reshape(reassign.shape[0], 1)))
                #assign new centroids as old
                cents = new_cents

        #return matrix with new centroids
        return mtx, new_cents

    def __kmeans_assign(self, mtx, centroids, k, dims):
        '''
        Given an unassigned matrix and some centroids, assign centroids to
        rows then perform kmeans
        '''

        #take matrix and centroids and assign to points
        assigned = self.__assign_pts(mtx, centroids, dims)

        #perform k_means until convergeance
        final_asg, final_cents = self.__kmeans_converge(assigned, centroids, k, dims)

        #return new assignments and centroids
        return final_asg, final_cents

    def __kmeans_group(self, data, k, convert=True, dm=1):
        '''
        perform kmeans. Convert Data into summary groups prior.
        data = matrix of points, either converted or not
        k = number of centroids
        convert = indicate whether to convert or not the data
        dims = dimensionality of the data. Fed to alg if
        '''

        #dimensionality set if working with groups already
        dims = dm
        mtx = data

        #conversion to "Triplice" Format if not already done
        #ignore if working with an already converted set
        if convert == True:
            #number of dimensions
            dims = data.shape[1]

            #data conversion
            mtx = self.__convertStoG(data)

        #initial assignment
        init_mtx, init_cents = self.__kmeans_groups_init(mtx, k, dims)

        #loop until convergeance
        final_asg, final_cents = self.__kmeans_converge(init_mtx, init_cents, k, dims)

        #return matrix with assignments as well as centroids
        return final_asg, final_cents

    def __nearest_cls(self, mtx, dims):
        '''
        given a matrix of assigned points/clusters, return the indices
        of the two nearest clusters
        '''

        nb_rows = mtx.shape[0]
        #minimum dist set to infinity
        min_distance = math.inf
        #placeholder indices
        min_inds = [-1, -1]

        #for each row
        for i in range(nb_rows):
            #for each pair of rows
            for j in range(i + 1, nb_rows):
                #get distance of these two
                current_dist = self.__dist_mean(mtx[i,:], mtx[j,:], dims)
                #current distance less than minimum
                if current_dist < min_distance:
                    #reset minimum
                    min_distance = current_dist
                    #set new minimum indices
                    min_inds = [i, j]

        return min_inds

    def __merge_clusters(self, mtx, merge_inds, group_nb):
        '''
        Given an assigned matrix and a list of indices to merge together,
        return a matrix with the specified rows merged together.
        group_nb = integer identifying the cluster used for the newly merged cluster
        assigned matrix: (SUM, SUMSQ, N, CLUSTER Nb.)
        '''

        #indices for rows to keep as is
        keep_inds = [x for x in list(range(mtx.shape[0])) if x not in merge_inds]

        #retrieve merging rows
        merging_rows = mtx[merge_inds,:]

        #retrieve rows to keep
        keep_rows = mtx[keep_inds,:]

        #sum rows
        merged = np.sum(merging_rows, 0)

        #replace last in index with group number
        merged[-1] = group_nb

        #re-add merged row to rest of dataset
        new_mtx = np.vstack((keep_rows, merged))

        return new_mtx

    def __assign_pts(self, mtx, cents, dims):
        '''
        given matrices for both points and centroids,
        assign to the points the nearest centroid.
        Return list of points with assigned centroids.
        '''

        #assign to each point closest centroid
        #column vector for assignments
        assignments = np.apply_along_axis(lambda x: self.__closest(x, cents, dims), 1, mtx)

        #matrix plus assignments
        assigned = np.hstack((mtx, assignments.reshape(assignments.shape[0], 1)))

        return assigned

    def __hierar_cls_agg(self, data, k):
        '''
        Perform  Agglomerative Hierarchical Clustering on the given Dataset, assigning to each point
        an individual cluster and progressively merging until k clusters is reached
        Return both assignments and the list of clusters
        '''

        #get number of dimensions
        dims = data.shape[1]

        #convert data to required format
        mtx = self.__convertStoG(data)
        #keep original matrix for later transformation
        mtx_init = self.__convertStoG(data)

        #initial assignment
        #list of clusters
        cluster_lst = np.arange(mtx.shape[0])
        #add to matrix
        mtx = np.hstack((mtx,cluster_lst.reshape(cluster_lst.shape[0], 1)))

        #while correct number of clusters has not been found
        while mtx.shape[0] != k:

            #get the two nearest rows
            near_ind = self.__nearest_cls(mtx[:,:-1], dims)

            #get cluster number of first row to merge
            cls_nb = mtx[near_ind[0],-1]

            #merge them together
            mtx = self.__merge_clusters(mtx, near_ind, cls_nb)

        #change matrix 'id's to just '0,1,2'
        mtx[:,-1] = np.arange(k)

        #assign points in original matrix to clusters
        assign_mtx = self.__assign_pts(mtx_init, mtx[:,], dims)

        return assign_mtx, mtx

    def __get_variances(self, row, dims):
        '''
        given a row, and number of dimensions, return the variance for
        each element.
        Return an array where elements are the variance for each element
        of the row.
        '''

        #sum
        row_sum = row[:dims]
        #sum of squares
        row_ssq = row[dims:dims * 2]
        #number of elements
        row_n = row[dims * 2]

        #variance
        variances = (row_ssq / row_n) - (((row_sum) ** 2) / (row_n ** 2))

        return variances

    def __mahalanobis_dist(self, row, centroids, dims):
        '''
        return a given element (singleton or otherwise)'s mahalanobis distance from 
        the given distribution. using centroid distance if the row in question is
        a collection of points summary
        '''

        #get point 
        point = row[:dims] / row[dims*2]

        #select from the list of centroids the distribution to use
        #row's assignment is currently closest centroid
        dist = centroids[int(row[-1]),:]

        #get dist avg
        dist_avg = dist[:dims] / dist[dims*2]
        #interval
        interval = point - dist_avg
        #get variances for distribution
        varis = self.__get_variances(dist, dims)
        #square interval and divide by vars
        int2 = (interval ** 2) / varis
        #sum and return distance
        return np.sum(int2)

    def __pc_merge(self, mtx):
        '''
        Merge operation used in primary compression.
        Given a submatrix, merge together all rows that have the same assignment
        '''

        #for each centroid ID

        c_ids = np.unique(mtx[:,-1])

        #print("Centroid IDs to merge")
        #print(c_ids)
        #print("Merge Matrix:")
        #print(mtx)

        for i in c_ids:
            #indices for the merge
            merge_inds = np.where(mtx[:,-1] == i)[0]

            #check if there are actually more than 1 row to merge
            #skip if not
            if len(merge_inds) > 1:

                mtx = self.__merge_clusters(mtx, merge_inds, i)

        #return newly formed matrix at the end
        return mtx

    def __primary_compression(self, mtx, centroids, dims, radius):
        '''
        Primary Compression step for the BFR clustering algorithm.
        mtx : matrix of assigned points
        centroids: current centroids
        dims: dimensionality
        radius: minimum mahalanobis distance under which points are compressed
        under the centroid
        '''

        #calculate mahalanobis distances for each point to the nearest centroid
        mh_dists = np.apply_along_axis(lambda x: self.__mahalanobis_dist(x, centroids, dims), 1, mtx)

        #convert NaN values to 0
        mh_dists = np.nan_to_num(mh_dists)

        #print("Mahalanobis Distances")
        #print(mh_dists)

        #check if distance is less than threshold
        #compress points to centroid if distance is less 
        threshold = mh_dists < radius
        
        #select rows to be compressed, this includes the centroids,
        #as their dist is 0
        compr_inds = np.where(threshold == True)[0]

        #separate matrix into 2: indices to be merged and 
        #those to be left alone

        #print("Full mtx: ", mtx.shape)

        #rows to merge
        to_merge = mtx[compr_inds,:]

        #rows to keep
        noCompr_inds = [x for x in list(range(mtx.shape[0])) if x not in compr_inds]

        to_leave = mtx[noCompr_inds,:]

        #print("To merge: ", to_merge.shape)
        #print("To keep: ", to_leave.shape)

        #merge selected indices, then append to kept
        merged = self.__pc_merge(to_merge)

        new_mtx = np.vstack((merged, to_leave))

        #print("Remade Matrix:")
        #print(new_mtx)

        return new_mtx

    def __get_tightness(self, row, dims):
        '''
        get tightness for given distribution, which is essentially
        the max of the standard deviations
        '''
        #get array of variances
        variances = self.__get_variances(row, dims)
        #square root is standard deviation
        st_div = variances ** (1 / 2)
        #get maximum value
        std_max = np.max(st_div)

        return std_max

    def __merged_tightness(self, row1, row2, dims):
        '''
        merge two rows together and get check their
        tightness
        '''

        row_c = row1 + row2

        tgt = self.__get_tightness(row_c, dims)

        return tgt

    def __get_tightest_pair(self, mtx, dims, orig_cts):
        '''
        given a matrix of rows of clusters, return the indices of the rows
        that are considered the "tightest", as well as the value
        ORIG_CTS: index values of original centroids, make sure not to merge
        groups that are both "REAL" clusters
        '''

        nb_rows = mtx.shape[0]
        #minimum dist set to infinity
        min_tightness = math.inf
        #placeholder indices
        min_inds = [-1, -1]

        #for each row
        for i in range(nb_rows):
            #for each pair of rows
            for j in range(i + 1, nb_rows):
                #get tightness from merged row of these two
                proj_tightness = self.__merged_tightness(mtx[i,:], mtx[j,:], dims)
                #current distance less than minimum
                if proj_tightness < min_tightness:
                    #check assigned indices are not BOTH part of original matrices
                    i1_check = i in orig_cts
                    i2_check = j in orig_cts
                    #only merge if at least one index is from a subgroup
                    if (i1_check and i2_check) == False:
                        #reset minimum
                        min_tightness = proj_tightness
                        #set new minimum indices
                        min_inds = [i, j]

        return min_inds, min_tightness

    def __hierar_cls_agg_tight(self, mtx, k, beta, dims, orig_cts):
        '''
        Perform  Agglomerative Hierarchical Clustering on the given matrix, assigning to each point
        an individual cluster and progressively merging the groups whose projected "tightness" is less
        than or equal to beta. Stop when no more merge options are available
        mtx: matrix of assigned points
        k: number of clusters
        beta: rows who are tighter than this value are merged
        dims: dimensions
        orig_cts: IDs of centroids from original Clustering Algorithm K. We want make sure we only 
        merge if one of the subclusters available is not already assigned to a "Main" cluster.
        '''

        #stopping condition
        stop_merge = False

        #while correct number of clusters has not been found
        while stop_merge == False:

            #get the two tightest row indices, and the value
            tight_ind, t_val = self.__get_tightest_pair(mtx, dims, orig_cts)

            #if the value is greater than beta, stop
            if t_val > beta:
                stop_merge = True
            else:
                #if value equal or less, merge and iterate again

                #get cluster number of first row to merge
                cls_nb1 = mtx[tight_ind[0],-1]
                #get ID for second row to merge
                cls_nb2 = mtx[tight_ind[1],-1]
                #take the minimum of the two
                cls_nb = min(cls_nb1, cls_nb2)

                #merge them together
                mtx = self.__merge_clusters(mtx, tight_ind, cls_nb)

        #return new matrix
        return mtx


    def __secondary_compression(self, mtx, centroids, dims, beta, k2):
        '''
        Secondary Compression Step. Take remaining singleton points and attempt
        to cluster them though kmeans. Find subclusters that are "tight", then
        merge them together through agglomerative hierarchical clustering while
        the tightness bound still holds.
        beta: tightness bound, standard deviation. Merge subclusters while they
        are still considered "tight".
        k2: number of clusters for subclustering, assumed k2 > K
        '''

        #separate singleton points from clusters
        
        #indices for singletons
        single_inds = np.where(mtx[:,dims*2] == 1)[0]
        #indices for clusters
        clust_inds = [x for x in list(range(mtx.shape[0])) if x not in single_inds]

        #separate
        #singleton elements
        singletons = mtx[single_inds,:]
        #clustered elements
        clustered_pts = mtx[clust_inds,:]
        
        #If the value of k2 is greater than the number of singletons,
        #skip secondary compression

        if k2 > singletons.shape[0]:
            return mtx

        #run kmeans on the singleton points with k2 > K
        #only if the number of singleton points exceeds k2

        subclusters, subcls_cts = self.__kmeans_group(singletons[:,:-1], k2, convert=False, dm=dims)

        #adjust IDs of subclusters so that they are not confounded with
        #"main" centroids

        #get number of centroids
        octs_nb = centroids.shape[0]

        #get IDs for the K centroids
        k1_ids = np.unique(centroids[:,-1])

        #adjust IDs
        subclusters[:,-1] += octs_nb
        subcls_cts[:,-1] += octs_nb

        #Identify "Tight" Subclusters

        #get tightness for subclusters
        sub_tight = np.apply_along_axis(lambda x: self.__get_tightness(x, dims), 1, subcls_cts)
        #identify "tight" subclusters
        #discard any over the threshold
        tight_thresh = sub_tight > beta
        #get indices
        tight_inds = np.where(tight_thresh == False)[0]
        #get the others
        loose_inds = np.where(tight_thresh == True)[0]

        #proceed if there are any tight subclusters
        if len(tight_inds) > 0:
            #slice
            tight_cls = subcls_cts[tight_inds,:]

            #add to list of clusters from earlier
            cls_plus_t = np.vstack((clustered_pts, tight_cls))

            #perform agglomerative hierarchical clustering on cls_plus_t
            cls_merged = self.__hierar_cls_agg_tight(cls_plus_t, k2, beta, dims, k1_ids)

            #slice loose centroids
            loose_cls = subcls_cts[loose_inds,:]

            #get remaining singletons that were not merged
            subc_nm = np.apply_along_axis(lambda x: x[-1] in loose_cls[:,-1], 1, subclusters)

            unmerged_inds = np.where(subc_nm == True)[0]

            #print('Unmerged inds:', unmerged)
            #print(unmerged_inds)

            #slice singleton list
            loose_singles = subclusters[unmerged_inds,:]

            #stack with centroids/tight clusters
            final_mtx = np.vstack((cls_merged, loose_singles))
            
        else:
            #no tight subclusters, just return original matrix
            final_mtx = mtx

        return final_mtx

    def __bfr_loop(self, data, centroids, k, dims):
        """
        The standard loop for the BFR algorithm:
        K-means, then Primary Compression, Then Secondary Compression
        K-means not done from scratch, points
        data: matrix of unassigned points, in cluster format
        centroids: matrix of points chosen as centroids
        """

        #assign data to centroids and perform k-means
        mtx_assign, new_cents = self.__kmeans_assign(data, centroids, self.__k, self.__dims)

        #primary compression
        compressed_asg = self.__primary_compression(mtx_assign, new_cents, self.__dims, self.__alpha)

        #secondary compression
        #k2 > K. Set to k2 = K * 2
        compressed2_asg = self.__secondary_compression(compressed_asg, new_cents, self.__dims, self.__beta, self.__k * 2)

        #return compressed matrix and new centroids
        return compressed2_asg, new_cents

    def __create_clusters(self, centroids, mtx):
        """
        Given a list of centroids and a matrix of assigned points, create cluster objects
        and store them
        """

        #for each centroid:
        for i in range(centroids.shape[0]):

            #create base cluster
            cluster_W = Cluster(None, None)
            #set center
            #take sum of points and divide by size of group
            cluster_W.center = list(centroids[i,:self.__dims] / centroids[i,-2])
            #add to list of clusters
            self.__clusters.append(cluster_W)
        #iterating through matrix to store values
        for i in range(mtx.shape[0]):
            #identify assigned centroid
            cent = mtx[i,-1]
            #get points
            point = list(mtx[i,:self.__dims])
            #add to the right cluster
            self.__clusters[int(cent)].points.append(point)
            self.__clusters[int(cent)].indexes.append(i)

    def get_clusters(self):
        """
        Return list of clusters
        """

        return self.__clusters

    def get_indexes(self):
        """
        Return list of Indexes
        """

        return [cluster.indexes for cluster in self.__clusters]

    def get_centres(self):
        """
        Return Cluster Centroids
        """

        return [cluster.center for cluster in self.__clusters]

    def cluster_noPart(self):
        """
        Cluster the given data without partitioning it. Essentially going through
        one cycle of KMEANS, PRIMARY COMPRESSION and SECONDARY COMPRESSION, then returning a result.
        """

        #begin by performing K-Means on the data

        data = self.__data

        self.__dims = data.shape[1]

        assignments_k, centroids_k = self.__kmeans_group(data, self.__k)

        #next, do primary compression

        compressed_asg = self.__primary_compression(assignments_k, centroids_k, self.__dims, self.__alpha)

        #next, secondary compression
        #using k2 = 2 * K for now.
        compressed2_asg = self.__secondary_compression(compressed_asg, centroids_k, self.__dims, self.__beta, self.__k * 2)

        #reassign centroids to points from secondary compr.
        reassigned = self.__assign_pts(compressed2_asg[:,:-1], centroids_k, self.__dims)

        #reassign until convergeance
        #get final centroids
        assignments_k, centroids_k = self.__kmeans_converge(reassigned, centroids_k, self.__k, self.__dims)

        #convert data to group format
        data_g = self.__convertStoG(self.__data)

        #assign to original points
        final_assign = self.__assign_pts(data_g, centroids_k, self.__dims)

        #create cluster objects
        self.__create_clusters(centroids_k, final_assign)

        #return final_assign, centroids_k

    def cluster_partition(self, filename, chunk_size, separator, k, alpha, beta, truths=True):
        """
        Read a dataset from file in chunks of the specified size.
        filename: name of file to read
        chunk_size: size of chunks to load into memory. in Bytes
        separator: separator for the file to read
        Then BFR arguments...
        truths : will ignore last column during clustering, assuming these are true values
        """
        #set params
        self.__data = None
        self.__k = k
        self.__alpha = alpha
        self.__beta = beta
    
        #list of Clusters
        self.__clusters = []

        #open the file
        f = open(filename, "r")

        #read the first line to determine number of rows / item size
        line1 = f.readline()
        #turn into numpy array
        #take 1 less column is truths values are read
        l1 = np.fromstring(line1, sep=separator) if truths == False else np.fromstring(line1, sep=separator)[:-1]
        #get memory size per line, size is size of item by number of columns
        chunk_line = l1.itemsize * l1.size
        #get number of columns
        nb_cols = l1.size

        #

        #Data Dimensionality
        self.__dims = nb_cols

        #get number of lines to load per partition
        lines_per_chunk = int(chunk_size / chunk_line)
        #check for end of file
        end_file = False
        #checks for first iteration exception
        first_iter = True
        f_iter2 = True
        #sum total of data to return
        data_total = np.zeros(nb_cols).reshape(1,nb_cols)
        #holding centroids
        cents = None

        while end_file == False:
            #until end of file is reached

            #array with dummy line, to be removed
            data_m = np.zeros(nb_cols).reshape(1,nb_cols)

            #First, read the next chunk of data into memory
            for i in range(lines_per_chunk):

                #check if this is the first iteration
                if first_iter == True:
                    #special case for first iteration
                    #add first line to matrix
                    data_m = np.vstack((data_m, l1))
                    #remove first iteration
                    first_iter = False
                else:
                    #normal execution
                    #read a line from the file
                    nline = f.readline()
                    #check line size
                    l_size = len(nline)
                    #if the string read is of length 0, end of file reached
                    if l_size == 0:
                        #mark end of file, break loop
                        end_file = True
                        break
                    #otherwise, continue
                    #convert to numpy array
                    line_a = np.fromstring(nline, sep=separator) if truths == False else np.fromstring(nline, sep=separator)[:-1]
                    #add to matrix
                    data_m = np.vstack((data_m, line_a))
        
            #loop complete

            #if resulting matrix had no rows added, stop loop
            if data_m.shape[0] == 1:
                break
            
            #remove dummy line
            data_m = data_m[1:,:]

            #otherwise, continue

            #convert data to summary format
            data_m = self.__convertStoG(data_m)

            #do operations here
            #add to totals
            if f_iter2 == True: #check for first iteration
                #keep the uncompressed data in memory for final assignment
                self.__data = data_m

                #set as total
                data_total = data_m

                #run K-means on the data, get assigned points and centroids
                #remove the assigned column from data_total
                data_total, cents = self.__kmeans_group(data_total, self.__k, convert=False, dm=self.__dims)

                #pass a loop of BFR
                data_total, cents = self.__bfr_loop(data_total[:,:-1], cents, self.__k, self.__dims)
                #drop point assignments
                data_total = data_total[:,:-1]

                #first iter done
                f_iter2 = False
            else:
                #add to uncompressed data
                self.__data = np.vstack((self.__data, data_m))

                #add to working data
                data_total = np.vstack((data_total, data_m))
                #perform BFR loop
                data_total, cents = self.__bfr_loop(data_total, cents, self.__k, self.__dims)
                #drop point assignments
                data_total = data_total[:,:-1]


        #close file
        f.close()

        #final assignment
        final_assign = self.__assign_pts(self.__data, cents, self.__dims)

        #convert to cluster object
        self.__create_clusters(cents, final_assign)

        #done
