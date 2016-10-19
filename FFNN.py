#!/usr/bin/env python

# CS430 Team Project 
# Date: May 9, 2015
# Team Members: Joe Dollard <jdollard@gmail.com>, Hamid Reza mohebbi <mohebbi.h@gmail.com>, Darian Springer <darianc.springer@gmail.com>

# FFNN (using pybrain library) and logistic regression parts

from os.path import isfile, join, splitext, basename, dirname
from sklearn import preprocessing
import sklearn.decomposition as skld
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import argparse
import csv
import numpy as np
import sys
import random
import matplotlib.image as mpli
import matplotlib.pyplot as plt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.networks  import FeedForwardNetwork
from pybrain.structure.connections import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import TanhLayer
from pybrain.structure.modules   import LinearLayer
from sklearn.linear_model import LogisticRegression


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
        min_val = min(cophenetic) 
        min_val = max( (min_val * .9, 0) )
        max_val = max(cophenetic)
        max_val = min( (max_val * 1.1), 1)
        plt.axis([rank[0], rank[len(rank)-1], min_val, max_val]) 
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

        plt.imshow(C)        
        plt.title('K=%s'%rank)        
        plt.savefig("%s_consensus_%d.png" % (name,rank))            
        plt.clf()

    # this method was created based on page 5 of 8 of the Park 2007 paper
    def dispersion(self, consensus_mtx):
        A = consensus_mtx
        n = A.shape[0]
        n_squared = n * n
        summation = 0
        for i in range(0, A.shape[0]):
            for j in range(0, A.shape[0]):
                summation = summation + (4 * ((A[i,j] - 0.5) * (A[i,j] - 0.5)) )

        dispersion_coefficient = (1.0/n_squared) * summation
        return dispersion_coefficient
    
    @classmethod
    def gen_data_matrix(cls, cluster_defs, overlaps, noise=0, rand_seed=None):
        """ Generate a random binary matrix that's constrained by the supplied paramaters:
            * cluster_defs: a list where each item is a dict that represents a single cluster.
                The dict must have keys 'features' (number of features for this cluster), and
                'samples' (number of samples for this cluster)
            * overlaps: an array that contains information about additional features that are
                shared across clusters.  Each item in the array should be a dict with keys:
                clusters: a list of the cluster indexes that share the feature
                num: number of shared features for the specified clusters
            * noise: a value between 0 and 1 which represents what percentage of the '1' values
                for each sample should be swapped with a non-one value on the row.  These swaps
                occur randomly
            * rand_seed: an optional number to seed the random number generator with.  Supports
                making the output matrix deterministic 

            Returns:
                a 2-tuple whose first value is a matrix, and whose second value is a list with
                the actual owning cluster index of each row in the matrix (i.e. ground truth)

            Example use: 
                x = Matrix.gen_data_matrix(
                    [ {'features': 5, 'samples': 5}, {'features': 5, 'samples': 10} ],
                    [ {'clusters': (0,1,), 'num': 1} ], noise=.1)

        """
        clusters = map(
            lambda c_def: {
                'idx': c_def[0],
                'features': c_def[1]['features'],
                'samples': c_def[1]['samples'],
                'cols': set(), # indicies of the columns/features
                'rows': set(), # indicies of the rows/samples
            }, enumerate(cluster_defs))

        num_features = sum( map(lambda x: x['features'], cluster_defs) )
        num_features += sum( map(lambda x: x['num'], overlaps) )
        num_samples = sum( map(lambda x: x['samples'], cluster_defs) )

        # optionally seed the random number for reproducable results
        random.seed(rand_seed)

        ### calculate what columns to assign to each cluster
        free_cols = range(0, num_features)
        random.shuffle(free_cols)

        # for each set of overlapping columns pick the cols to use, then add them
        # to the specified clusters
        for overlap in overlaps:
            overlap_cols = free_cols[:overlap['num']] 
            free_cols = free_cols[overlap['num']:]

            for cluster_idx in overlap['clusters']:
                clusters[cluster_idx]['cols'].update(overlap_cols)

        # pick the remaining columns for each cluster
        for idx, cluster in enumerate(clusters):
            take_cols = cluster['features']
            cluster['cols'].update( free_cols[:take_cols] )
            free_cols = free_cols[take_cols:]

        ### calculate what rows (i.e. samples) to assign to each cluster
        free_rows = range(0,num_samples)
        random.shuffle(free_rows)

        for idx, cluster in enumerate(clusters):
            take_rows = cluster['samples']
            cluster['rows'] = free_rows[:take_rows]
            free_rows = free_rows[take_rows:]

        ### generate the matrix
        mtx = np.zeros( (num_samples, num_features) )
        cluster_membership = list()

        for row_idx in range(0,num_samples):
            r_clusters = filter(lambda c: row_idx in c['rows'], clusters)

            owner = r_clusters[0]
            cluster_membership.append(owner['idx'])

            mtx[row_idx] = cls._gen_row(
                row_len=num_features, ones=r_clusters[0]['cols'], noise=int(noise*num_features))

        return (mtx, cluster_membership)

    @classmethod
    def _gen_row(cls, row_len, ones, noise):
        """ generate a row of a matrix.  Arguments:
            row_len: the number of columns in the row
            ones: a set of columns indicies that should be ones
            noise: the number of columns whose entries should be 1, that we'll move to a non-one
                columns"""
        row = np.zeros(row_len)

        # create noiseless row
        for i in range(0,row_len):
            if i in ones:
                row[i] = 1
        
        # add noise
        c_ones = list(ones)
        zeros = [x not in c_ones for x in range(0,row_len)]

        random.shuffle(c_ones)
        random.shuffle(zeros)

        for i in range(0,min(noise, len(zeros), len(ones))):
            row[ zeros[i] ] = 1
            row[ c_ones[i] ] = 0

        return row

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

    def get_consensus_matrix(self, max_diff=0.01, min_runs=250, max_runs=500):
        """ Algorithm here is adapted from Brunet & Tamayo:
            repeat until consensus matrix appears to stabilize:
                Run NMF on the data
                Determine cluster membership by taking the biggest value in each column of H in the NMF decomposition
                Create an m*m connectivity matrix based on the cluster membership
                Merge the connectivity matrix with our concensus matrix, with is the average connectivity matrix over our runs """

        num_runs, curr_diff = (0, 1)
        consensus_mtx = np.zeros( (len(self._matrix[0]), len(self._matrix[0])) )

        # we retain all w & h matricies, then return the one whose connectivity matrix has
        # the smallest difference to the consensus matrix
        all_decomps = list()

        while num_runs <= max_runs and (num_runs < min_runs or curr_diff >= max_diff): 
            sys.stdout.write('.')
            (n_mtx, w, h) = self.get_connectivity_matrix()
            all_decomps.append( {'conn_mtx':n_mtx, 'w': w, 'h': h} )

            n_consensus_mtx = consensus_mtx + n_mtx
            if num_runs > 0:
                curr_diff = self._diff(
                    consensus_mtx / (num_runs + 0.0),
                    n_consensus_mtx / (num_runs + 1.0) )

            consensus_mtx = n_consensus_mtx
            num_runs += 1

        consensus_mtx = consensus_mtx / num_runs
        print("\nFinished %d rounds of nmf" % num_runs)

        # find the connectivity matrix w/ the smallest diff
        min_decomp = min(all_decomps,
            key=lambda x: self._diff(consensus_mtx, x['conn_mtx']))

        return (consensus_mtx, min_decomp['w'], min_decomp['h'])

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

        model = skld.NMF(
            n_components=self._num_clusters,
            init='random', beta=.3, eta=.5, max_iter=5000, nls_max_iter=10000 )

        w = model.fit_transform(self._matrix)
        h = model.components_

        # convert the 'H' matrix, which represents weights for our data matrix W, into
        # an array representing cluster membership. Index of biggest value in each 
        # col of matrix H is the cluster
        clusters = []
        model_width = len(h[0])

        for col_idx in range(model_width):
            # convert column into an array
            col_vals = h[:,col_idx]

            # determine biggest row index and it's value from the array
            (row_idx,max_val) = max( enumerate(col_vals), key=lambda x: x[1] )

            clusters.append(row_idx)

        # clusters array, w, h
        return (clusters, w, h)

    def _get_clustering_matrix(self, clusters):
        """ convert from an array to a one-hot matrix representation of cluters """
        clustering_mtx = np.zeros( (len(clusters), self._num_clusters) )
        for i in range( len(clusters) ):
            val = clusters[i]
            clustering_mtx[i][val] = 1
        return clustering_mtx

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
    parser.add_argument("class_file", help="path to data file (in CSV format)")
    #parser.add_argument("out_dir", help="output directory")

    args = parser.parse_args()

    if not isfile(args.data_file):
        print("file %s does not exist" % args.data_file)
        sys.exit(1)
    
    if not isfile(args.class_file):
        print("file %s does not exist" % args.class_file)
        sys.exit(1)

    """if isfile(args.out_dir):
        print("output directory '%s' is actually a file" % args.out_dir)
        sys.exit(1)
    """
    return args

def partition_samples(mtx, classes, test_set_ratio=0.1):
    """ Partition a matrix into two matricies; one test set and one training set.  Returns
        two tuples: ( (training_mtx, training_classes), (test_mtx, test_classes) ). """
    training_rows = list()
    training_classes = list()
    training_counts = dict()
    test_rows = list()
    test_classes = list()
    test_counts = dict()

    if (len(classes) != len(mtx)):
        raise Exception("invalid args: incorrect number of classes specified: %d vs %d" % (len(classes), len(mtx)))

    for idx,row in enumerate(mtx):
        # if we haven't met our ratio, use for test set.  Otherwise use for training set.
        sample_class = classes[idx]
        curr_ratio = ((test_counts.get(sample_class, 0) + 0.0) /
            (training_counts.get(sample_class, 1) + test_counts.get(sample_class, 0)) + 0.0)
        if curr_ratio < test_set_ratio:
            test_counts[sample_class] = test_counts.get(sample_class, 0) + 1
            test_rows.append(row)
            test_classes.append(sample_class)
        else:
            training_counts[sample_class] = training_counts.get(sample_class, 0) + 1
            training_rows.append(row)
            training_classes.append(sample_class)

    # convert our lists into matrices
    training_mtx = np.zeros( (len(training_classes), len(mtx[0])) )
    for idx,row in enumerate(training_rows):
        training_mtx[idx] = row

    test_mtx = np.zeros( (len(test_classes), len(mtx[0])) )
    for idx,row in enumerate(test_classes):
        test_mtx[idx] = row
    
    return {
        'training': {'matrix': training_mtx, 'classes': training_classes},
        'test': {'matrix': test_mtx, 'classes': test_classes}
    }
    
def logistic_regression(mtx, classes, use_distinct_test_set=True, test_set_ratio=0.1): 
    # determine the class that each sample is in    
    if use_distinct_test_set:
        data = partition_samples(mtx, classes, test_set_ratio)
        #print("--> training size: %d" % len(data['training']['classes']))
        #print("--> test size: %d" % len(data['test']['classes']))
        l_reg = LogisticRegression(multi_class='ovr',tol=0.1, C=10)
        l_reg.fit(data['training']['matrix'], data['training']['classes'])
        return l_reg.score(data['test']['matrix'], data['test']['classes'])
    else:
        l_reg = LogisticRegression()
        y = l_reg.fit(mtx, classes)
        return l_reg.score(mtx,classes)

def main():
    results = []
    args = parse_args()

    # there's a bug in the _sparseness method in sklearn's nmf module that is
    # hit in some edge cases.  The value it computes isn't actually needed in
    # this case, so we can just ignore this divide by 0 error
    np.seterr(invalid="ignore")

    mtx = np.loadtxt(args.data_file, delimiter=',',skiprows=1)      
    clabels = np.loadtxt(args.class_file, delimiter=',')     

    print("Matrix is %d by %d and %f sparse" % (len(mtx), len(mtx[0]), Matrix.get_sparsity(mtx)))
    #print("clabels is %d by %d and %f sparse" % (len(clabels), len(clabels[0]), Matrix.get_sparsity(clabels)))
    #mtx = np.matrix.transpose(mtx)  # transpose to put samples into columns, genes into rows   

    # create random class labels, replace with result of NMF
    #clabels = np.zeros(len(mtx))
    #for i in range(len(mtx)):
       # clabels[i] = random.randint(0, 3)
    clabels = np.matrix.transpose(clabels) 
        
    print '-----------Logestic Regression-----------'
    t_lacc =0    
    for i in range(10):        
        t_lacc = t_lacc + logistic_regression(mtx,clabels,True)
        
    print 'accuracy of logistic regression ', (t_lacc * 10)
    
    
    print '-----------ANN Computation----------'
    # prepare dataset for ANN
    ds = ClassificationDataSet(len(mtx[0]), 1 , nb_classes=5) # replace with result of NMF
    for k in xrange(len(mtx)): 
        ds.addSample(np.ravel(mtx[k]),clabels[k])
    
    # 10-fold cv    
    t_error =0;
    t_acc =0;
    for i in range(10):    
        # divide the data into training and test sets        
    
        tstdata_temp, trndata_temp = ds.splitWithProportion(0.10)
        
        tstdata = ClassificationDataSet(len(mtx[0]), 1 , nb_classes=5)
        for n in xrange(0, tstdata_temp.getLength()):
            tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
            
        trndata = ClassificationDataSet(len(mtx[0]), 1 , nb_classes=5)
        for n in xrange(0, trndata_temp.getLength()):
            trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
        
        trndata._convertToOneOfMany( )
        tstdata._convertToOneOfMany( )
        
        fnn = FeedForwardNetwork()
        inp = LinearLayer(trndata.indim)
        h1 = SigmoidLayer(10)
        h2 = TanhLayer(10)
        h3 = TanhLayer(10)
        h4 = TanhLayer(10)
        h5 = TanhLayer(10)
        outp = LinearLayer(trndata.outdim)
        #fnn = buildNetwork( trndata.indim, 10 , trndata.outdim, outclass=SoftmaxLayer )
        
        # add modules
        fnn.addOutputModule(outp)
        fnn.addInputModule(inp)
        fnn.addModule(h1)
        fnn.addModule(h2)
        fnn.addModule(h3)
        fnn.addModule(h4)
        fnn.addModule(h5)
        # create connections
        fnn.addConnection(FullConnection(inp, h1))
        fnn.addConnection(FullConnection(inp, h2))
        fnn.addConnection(FullConnection(inp, h3))
        fnn.addConnection(FullConnection(inp, h4))
        fnn.addConnection(FullConnection(inp, h5))
        fnn.addConnection(FullConnection(h1, h2))        
        fnn.addConnection(FullConnection(h2, h3))
        fnn.addConnection(FullConnection(h3, h4))
        fnn.addConnection(FullConnection(h4, h5))
                
        fnn.addConnection(FullConnection(h5, outp))
        
        fnn.sortModules()

        trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01) 
        
        #trainer.trainUntilConvergence()
        trainer.trainEpochs(5)
        
        t_error = t_error + percentError( trainer.testOnClassData (dataset=tstdata ), tstdata['class'] )
        
                
    
    print 'avg error ', (t_error/10)
    print 'avg acc ', (100- (t-error/10))
    

main()

