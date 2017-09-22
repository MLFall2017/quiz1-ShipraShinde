# Name: Shipra Shivaji Shinde
# ID: 800974877
#
# Principal Component Analysis (PCA)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def perform_pca(file_path):
    # read the dataset from csv file
    X = np.genfromtxt (file_path, delimiter=",")
    # delete the first row
    X = np.delete(X, (0), axis=0)

    X_std = StandardScaler().fit_transform(X)

    # Step 1: calculate mean center for all the columns
    mean_vec = np.mean(X_std, axis=0)

    # Step 2: calculate cov(x)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)

    # Step 3: calculate eigen values and eigen vectors of cov(x)
    eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' %eigen_vectors)
    print('\nEigenvalues \n%s' %eigen_values)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Step 4: Projection
    matrix_w = np.hstack((eig_pairs[0][1].reshape(X_std.shape[1],1),
                          eig_pairs[1][1].reshape(X_std.shape[1],1)))
    print('Matrix W:\n', matrix_w)
    Y = X_std.dot(matrix_w)
    print('Matrix Y:\n', Y)

    # Step 5: Plot the projections for the first and second principal components
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(Y[:,0],Y[:,1])
    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    fig.show()