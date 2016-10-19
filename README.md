# Genetic-Mutation-Discovery

This is my course project for the Applied Machine Learning (CS697) course during my PhD study at UMass Boston. The goal of this project is to discover genetic mutation patterns for specific tumor types using unsupervised learning method. The data of this project are real cancer data collected at Dana Farber and due to the copyright I couldn't put the data here online. This project implemented using Python language and its powerful libraries. 

In this project we used matrix factorization technique descibed by Brunet in his paper entitled "Metagenes and molecular pattern discovery using matrix factorization". Using Nonnegative Matrix Factorization (NMF) we reduced the dimentionality of the data and increased the classification accuracy. 

At first step, we used PCA algorithm to select most important features and we used logical regression for classification (NMF-LogicalReg.py). 

The second step contains using Feed Forward Neural Network for classification. The Neural Network contains 5 layers (FFNN.py)

The files are organized as follow:

1- NMF-LogicalReg.py : This file contains the code for NMF and logical regression method.
2- FFNN.py: The FFNN code that use Pybrain Python library.
3- results: This folder contains Reordered consensus matrices of the cancer data and also some of our results.

