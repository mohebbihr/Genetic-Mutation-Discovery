#!/usr/bin/env python3

# CS697 Team Project
# Date: April 9, 2015
# Team Members: Joe Dollard <jdollard@gmail.com>, Hamid Reza mohebbi <mohebbi.h@gmail.com>,, Darian Springer <darianc.springer@gmail.com>
# Using NMF, PCA for noise, dimension reduction and Logistic Regression for classification

from os.path import isfile, join, splitext, basename, dirname
from sklearn import preprocessing
from sklearn.decomposition import ProjectedGradientNMF
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import argparse
import csv
import numpy as np
import sys
import random
import math
import matplotlib.image as mpli
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection)

class Matrix:
    @staticmethod
    def to_binary(matrix):
        """ convert all non-zero values in a matrix to 1 """
        for i in range(0,len(matrix)):
            for j in range(0,len(matrix[i])):
                if matrix[i][j] > 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        return matrix

    @staticmethod
    def dupe_rows(matrix, n):
        """ for each row in an input matrix, make an output matrix that has each row repeated
            n times """
        v_matrix = matrix
        #v_matrix = np.matrix.view(matrix, np.ndarray)
        n_matrix = np.zeros( (len(v_matrix) * n, len(v_matrix[0])) )

        for i in range(0,len(v_matrix)):
            for j in range(0,len(v_matrix[i])):
                for k in range(0,n):
                    n_matrix[ (i * n + k) ][j] = v_matrix[i][j]

        return n_matrix

    @staticmethod
    def get_sparsity(mtx):
        height = len(mtx)
        width = len(mtx[0])
        count = 0
        
        for i in range(height):
            for j in range(width):
                if mtx[i][j] == 0:
                    count += 1
        return count / (height * width)

    @classmethod
    def trim_cols(cls, matrix, threshold=100, silent=False):
        """ remove all columns that are consist of only 0's threshold % of the time """
        num_cols = len(matrix[0])
        matrix, num_trimmed = cls._trim_cols(matrix, threshold)
        print("discarded %d cols from %d total" % (num_trimmed, num_cols))
        return matrix

    @classmethod
    def trim_rows(cls, matrix, threshold=100):
        """ remove all rows that are consist of only 0's threshold % of the time """
        t_trimmed, num_trimmed = cls._trim_cols( np.matrix.transpose(matrix), threshold )
        print("discarded %d rows" % num_trimmed)
        return np.matrix.transpose( t_trimmed )

    @staticmethod
    def to_binary_max_val(matrix):
        """ Mark the biggest element in each row with a one, and the rest with 0 """
        for i in range(0,len(matrix)):
            max_val = -1
            max_idx = -1
            for j in range(0,len(matrix[i])):
                if matrix[i][j] > max_val:
                    max_val = matrix[i][j]
                    max_idx = j
                matrix[i][j] = 0
            matrix[i][max_idx] = 1
        return matrix

    @staticmethod
    def reorder(matrix):
        # use average linkage HC to reorder the samples and thus the rows and columns of C
        # This method came from: http://nimfa.biolab.si/nimfa.examples.all_aml.html      
        Y = 1 - matrix
        Z = linkage(Y, method='average')
        ivl = leaves_list(Z)
        ivl = ivl[::-1]
        return matrix[:, ivl][ivl, :]

    # sort our 'h' matrix such that columns that are in the same cluster are next
    # to each other
    @staticmethod
    def sort_cols_by_max_row(mtx):
        # for each column, create a hash with it's idx and row idx with the biggest value
        # sort this array of hashes by row idx
        # build up a new matrix w/ the columns from each col_idx item in our sorted array
        max_row_idxes = mtx.argmax(0)       # biggest row idx for each col
        max_cols = map(
            lambda x: {'col_idx': x[0], 'max_row_idx': x[1]},
            enumerate(max_row_idxes) )

        s_max_cols = sorted(max_cols, key = lambda x: x['max_row_idx'])

        sorted_mtx = np.zeros( (len(mtx), len(mtx[0])) )

        for new_col_idx,s_col in enumerate(s_max_cols):
            sorted_mtx[:,new_col_idx] = mtx[:,s_col['col_idx']]

        return sorted_mtx

    @staticmethod
    def to_heat_map(matrix, filename, min_width=200, min_height=200):
        # lengthen or widen the matrix if needed to make it easier to see on a graph
        if len(matrix) < min_height:
            matrix = Matrix.dupe_rows( matrix, int(min_height / len(matrix)) )
        if len(matrix[0]) < min_width:
            matrix_t = np.matrix.transpose(matrix)
            matrix_t = Matrix.dupe_rows( matrix_t, int(min_width / len(matrix_t)) )
            matrix = np.matrix.transpose(matrix_t)

        # 'hot' colormap for high contrast (see http://matplotlib.org/users/colormaps.html)
        mpli.imsave(filename, matrix, cmap=mpli.cm.GnBu)       # best

    @staticmethod
    def to_line_plot(matrix, filename):
        """ convert each row of a matrix into a continuous line on a plot.  Used to
        create a chart like the 'metagene expression profile' """
        plt.figure()
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        for idx,row in enumerate(matrix):
            plt.plot(row, label="row %d" % idx)

        plt.savefig(filename)
        plt.clf()

    @staticmethod
    def to_cophenetic_plot(rank,cophenetic, filename):
        plt.figure()
        plt.plot(rank, cophenetic)
        plt.axis([rank[0], rank[len(rank)-1], 0, 1]) 
        plt.xticks(rank)
        plt.xlabel('Rank K')
        plt.ylabel('Cophenetic correlation')
        plt.savefig(filename)   
        plt.clf()

    @staticmethod
    def to_consensus_plot(C, rank,name):
        """
        Plot reordered consensus matrix.
    
        :param C: Reordered consensus matrix.
        :type C: numpy.ndarray`
        :param rank: Factorization rank.
        :type rank: `int`
        """
        plt.set_cmap("RdBu_r")

        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.imshow(C)        
        plt.title('K=%s'%rank)        
        plt.savefig("%s_consensus_%d.png" % (name,rank))            
        plt.clf()

    @classmethod
    def _trim_cols(cls, matrix, threshold=100):
        """ remove all columns that are consist of only 0's threshold % of the time """

        # for each column, determine how many rows are non-zero
        num_non_zero = [ 0 for i in range(len(matrix[0])) ]
        num_rows, num_cols = len(matrix), len(matrix[0])

        for i in range(0,num_rows):
            for j in range(0,num_cols):
                if matrix[i][j] != 0:
                    num_non_zero[j] += 1

        # delete empty cols
        num_deleted = 0

        for col_idx,count in enumerate(num_non_zero):
            if ((num_rows - num_non_zero[col_idx]) / num_rows) * 100 >= threshold:
                matrix = np.delete(matrix,col_idx - num_deleted,1)
                num_deleted += 1

        return matrix, num_deleted

class Nmf:

    def __init__(self, matrix, clusters):
        """ The input matrix should have rows to represent features, and columns
            to represent samples """
        self._matrix = matrix
        self._num_clusters = clusters

    # this method was copied from the python nimfa library - http://nimfa.biolab.si/
    def coph_cor(self, consensus_mtx):
        A = consensus_mtx
        # upper diagonal elements of consensus
        avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                        for j in range(i + 1, A.shape[1])])
        # consensus entries are similarities, conversion to distances
        Y = 1 - avec
        Z = linkage(Y, method='average')
        # cophenetic correlation coefficient of a hierarchical clustering
        # defined by the linkage matrix Z and matrix Y from which Z was
        # generated
        return cophenet(Z, Y)[0]

    def get_consensus_matrix(self, max_diff=0.01, min_runs=10, max_runs=100):
        """ Algorithm here is adapted from Brunet & Tamayo:
            repeat until consensus matrix appears to stabilize:
                Run NMF on the data
                Determine cluster membership by taking the biggest value in each column of H in the NMF decomposition
                Create an m*m connectivity matrix based on the cluster membership
                Merge the connectivity matrix with our concensus matrix, with is the average connectivity matrix over our runs """

        num_runs, curr_diff = (0, 1)
        consensus_mtx = None

        # we retain all w & h matricies, then return the one whose connectivity matrix has
        # the smallest difference to the consensus matrix
        all_decomps = list()

        while num_runs <= max_runs and (num_runs < min_runs or curr_diff >= max_diff): 
            #print('.',end='',flush=True)
            (n_mtx, w, h) = self.get_connectivity_matrix()
            all_decomps.append( {'conn_mtx':n_mtx, 'w': w, 'h': h} )

            if consensus_mtx is None:
                consensus_mtx = n_mtx
            else:
                n_consensus_mtx = self._avg(consensus_mtx, n_mtx, num_runs)
                curr_diff = self._diff(consensus_mtx, n_consensus_mtx)
                consensus_mtx = n_consensus_mtx

            num_runs += 1

        print("\nFinished %d rounds of nmf" % num_runs)

        # find the connectivity matrix w/ the smallest diff
        min_decomp = None
        for decomp in all_decomps:
            diff = self._diff(consensus_mtx, decomp['conn_mtx'])

            if min_decomp is None or min_decomp['diff'] > diff:
                min_decomp = decomp
                min_decomp['diff'] = diff

        return (consensus_mtx, w, h)

    def get_connectivity_matrix(self):
        """ Returns a connectivity matrix, along with the w and h values that it was
            build from.  There are likely multiple possible connectivity matricies
            due to random numbers being used to initialize NMF """
        (clusters, w, h) = self.get_cluster_membership()
        cluster_mtx = self._get_clustering_matrix(clusters)
        conn_mtx = np.dot(cluster_mtx, np.matrix.transpose(cluster_mtx)) 
        return (conn_mtx, w, h)

    def get_cluster_membership(self):
        """ Determine the cluster number that each sample is associated with. """ 

        model = ProjectedGradientNMF(
            n_components=self._num_clusters,
            init='random', beta=.3, eta=.5, max_iter=5000 )

        w = model.fit_transform(self._matrix)
        h = model.components_

        # convert the 'H' matrix, which represents weights for our data matrix W, into
        # an array representing cluster membership. Index of biggest value in each 
        # col of matrix H is the cluster
        clusters = []
        model_width = len(h[0])

        for col_idx in range(model_width):
            max_val = dict()
            for row_idx in range(self._num_clusters):
                h_val = h[row_idx][col_idx]

                if not max_val or h_val > max_val['val']:
                    max_val = {'row_idx': row_idx, 'val': h_val}

            clusters.append(max_val['row_idx'])

        # clusters array, w, h
        return (clusters, w, h)

    def _get_clustering_matrix(self, clusters):
        """ convert from an array to a one-hot matrix representation of cluters """
        clustering_mtx = np.zeros( (len(clusters), self._num_clusters) )
        for i in range( len(clusters) ):
            val = clusters[i]
            clustering_mtx[i][val] = 1
        return clustering_mtx

    def _avg(self, mtx1, mtx2, weight1=1, weight2=1):
        """ Given two matricies of equal dimensions, calculate a new matrix that
            is the weighted average of the two """
        num_rows, num_cols = len(mtx1), len(mtx1[0])
        avg_mtx = np.zeros( (num_rows, num_cols) )

        for i in range(num_rows):
            for j in range(num_cols):
                w_sum = (mtx1[i][j] * weight1) + (mtx2[i][j] * weight2)
                avg_mtx[i][j] = w_sum / (weight1 + weight2)

        return avg_mtx

    def _diff(self, matrix1, matrix2):
        """ Given two matricies of the same dimensions, determine how different they are.
        Each cell in each matrix must be between 0 or 1.  We determine the difference by 
        calculating how much each cell has changed (in absolute terms), then adding up these
        numbers and dividing by the total number of cells.  This gives us a change ratio for
        the whole matrix.  0 = unchanged, 1 = changed the max possible """
        diff_sum = 0
        num_rows, num_cols = len(matrix1), len(matrix1[0])

        for i in range(num_rows):
            for j in range(num_cols):
                diff_sum += abs(matrix1[i][j] - matrix2[i][j])

        return diff_sum / (num_rows * num_cols)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="path to data file (in CSV format)")
    parser.add_argument("out_dir", help="output directory")

    args = parser.parse_args()

    if not isfile(args.data_file):
        print("file %s does not exist" % args.data_file)
        sys.exit(1)

    if isfile(args.out_dir):
        print("output directory '%s' is actually a file" % args.out_dir)
        sys.exit(1)

    return args

def main():
    results = []
    args = parse_args()

    # there's a bug in the _sparseness method in sklearn's nmf module that is
    # hit in some edge cases.  The value it computes isn't actually needed in
    # this case, so we can just ignore this divide by 0 error
    np.seterr(invalid="ignore")
    print("Processing %s" % args.data_file)
    full_mtx = np.loadtxt(args.data_file, delimiter=',')
    # normalize / clean-up our matrix: remove genes that are always 0, and
    # convert all non-zero values to 1
    min_col_mutation_pct = 1.5       # columns with < this percentage of mutations will be deleted
    full_mtx = Matrix.trim_cols(full_mtx, 100 - min_col_mutation_pct)
    full_mtx = Matrix.to_binary(full_mtx)    

    partitions = (
        {'start_row':0,   'end_row':269, 'name':"COAD"},
        {'start_row':270, 'end_row':689, 'name':"KIRC"},
        {'start_row':690, 'end_row':941, 'name':"PRAD"},
        {'start_row':942, 'end_row':1284,'name':"SKCM"},
        {'start_row':1285,'end_row':1341,'name':"UCS"} )

    # create an array with the classes of each sample
    classes = list()
    for partition_idx,partition in enumerate(partitions):
        for row_idx in range(partition['start_row'], partition['end_row'] + 1):
            classes.append(partition_idx)

    l_reg = LogisticRegression()
    lreg_mtx=l_reg.fit_transform(full_mtx, classes)
    accuracy = l_reg.score(full_mtx,classes)
    print("accuracy = %f" % accuracy) 
    print("regression dimensions : %d, %d"% (len(lreg_mtx),len(lreg_mtx[0])))   
    
    print("Computing Sparse PCA projection")
    pca_mtx = decomposition.SparsePCA(n_components=100).fit_transform(lreg_mtx)
    pca_mtx = Matrix.to_binary(pca_mtx)
    #rank_range = range(2,3)
    rank_range = range(2,11)

    for partition in partitions:
        print("Processing %s cancer" % partition['name']) 
        mtx = pca_mtx[ partition['start_row'] : partition['end_row'] + 1 ]
        mtx = Matrix.trim_rows(mtx)     # remove empty rows
        print("Matrix is %d by %d and %f sparse" % (len(mtx), len(mtx[0]), Matrix.get_sparsity(mtx)))
        mtx = np.matrix.transpose(mtx)  # transpose to put samples into columns, genes into rows

        print("=====> Finding the optimum # clusters (range = %d to %d) <=====" %
            (rank_range[0], rank_range[-1]) )

        results = list()

        for num_clusters in rank_range:
            print("Trying cluster size %d " % num_clusters)
            nmf = Nmf(mtx, clusters=num_clusters)
            c,w,h = nmf.get_consensus_matrix()
            coph = nmf.coph_cor(c)
            results.append(
                {'rank': num_clusters,
                 'consensus': Matrix.reorder(c),
                 'w': w,
                 'h': h,
                 'coph': coph })

        best = max( results, key=lambda x: x['coph'] )
        worst = min( results, key=lambda x: x['coph'] )

        print("worst: rank %d with %f" % ( worst['rank'], worst['coph']))
        print("best: rank %d with %f" % ( best['rank'], best['coph']))
        print("all:", [ x['coph'] for x in results ])

        base_name = join(args.out_dir, partition['name'])

        Matrix.to_cophenetic_plot( rank_range,
            [ x['coph'] for x in results ], base_name + "_cophenetic.png")        

        Matrix.to_heat_map(mtx, base_name + "_a.png")
        Matrix.to_heat_map(best['w'], base_name + "_w.png")
        Matrix.to_heat_map(best['h'], base_name + "_h.png")
        #sort h matrix
        ordered_h = np.sort(best['h'], axis = 0)
        Matrix.to_line_plot(ordered_h, base_name + "_h_line.png")
        Matrix.to_consensus_plot(best['consensus'], best['rank'], base_name)  	
        Matrix.to_consensus_plot(results[0]['consensus'], results[0]['rank'], base_name)
        Matrix.to_consensus_plot(results[1]['consensus'], results[1]['rank'], base_name)
        Matrix.to_consensus_plot(results[2]['consensus'], results[2]['rank'], base_name)
        Matrix.to_consensus_plot(results[3]['consensus'], results[3]['rank'], base_name)

main()



