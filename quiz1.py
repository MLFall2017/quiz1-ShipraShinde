# Name: Shipra Shivaji Shinde
# ID: 800974877

# Quiz 1

import numpy as np
import pca_module as pca
import os

os.chdir("E:\\UNCC\\Fall 2017\\Machine Learning\\\Repository\\quiz1-ShipraShinde")
# Write your Python script to:
#  calculate the variance of every variable in the data le.
#  calculate the covariance between x and y, and between y and z
#  do PCA of all the data in the given data le using your own PCA module.
#
#  read the dataset from csv file
X = np.genfromtxt ("dataset_1.csv", delimiter=",")
# delete the first row
X = np.delete(X, (0), axis=0)

# calculate the variance of every variable in the data file.
variance_vec = np.var(X, axis=0)
print('\nVariance of variables \n%s' %variance_vec)

# calculate the covariance between x and y, and between y and z
covariance_xy = np.cov(X[:,0], X[:,1])
covariance_yz = np.cov(X[:,1], X[:,2])
print('\nCovariance between x and y \n%s' %covariance_xy)
print('\nCovariance between y and z \n%s' %covariance_yz)

# do PCA of all the data in the given data fille using your own PCA module
Y = pca.perform_pca("dataset_1.csv")

# Question 3.2
a = np.array([[0, -1],[2,3]], dtype=float)
eigen_values, eigen_vectors = np.linalg.eig(a)
print('Eigenvectors \n%s' % eigen_vectors)
print('\nEigenvalues \n%s' % eigen_values)