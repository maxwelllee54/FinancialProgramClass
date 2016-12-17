###############################################################
## Quiz5_Q2.py
## Update cisc5352.lecture.8.demoKNNSVMIVPrecition.py such that it can do kernel map approximation by using all kernels
## Original Author: Henry Han
## Last update: Nov 30, 2016
################################################################

# Import packages

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import numpy as np
import csv
from datetime import datetime


######################################
## Define functions
#######################################

## Mean square error: MSE
def get_MSE(Error):
    mse=np.sum(np.power(Error,2))
    mse=mse/len(Error)
    return mse



######################################
##  Read data from CSV file
#######################################


filename='Option_Data_2000.csv' # aka 'cisc5352.project.1.option_data.csv' first 2000 features
Data    = csv.reader(open(filename, 'r'))

# Save the original data into list
Data_input    = []
Data_response = []

# Read the data from csv file
# response variable: implied volaility

for row in Data:
    Data_response.append(row[7])  # implied volaility
    row.pop(7)
    Data_input.append(row)

# Delete the header information in data and response variables
Data_response.pop(0)
Data_input.pop(0)


# Convert the data from string to float: a wordy approach
Data_input_convert    = []
Data_response_convert = []

for each_row in Data_input:
    temp = []
    for entry in each_row:
        temp.append(float(entry))
    Data_input_convert.append(temp)

for response_variable_entry in Data_response:
    Data_response_convert.append(float(response_variable_entry))

Data_input    = Data_input_convert
Data_response = Data_response_convert



# Pick only first N samples
N             = 1000    # large N can lead a slow SVM!
Data_input    = Data_input[0:N]
Data_response = Data_response[0:N]

# Split the data set into training data(80%) and testing data(20%)

Input_train, Input_test, Response_train, Response_test = \
    train_test_split(Data_input, Data_response, test_size=0.2, random_state=42)


# Set kernel list

kNN_algo_list = ['auto', 'ball_tree', 'kd_tree', 'brute']

for algo in kNN_algo_list:

    print("\nkNN model is running under algorithm: {:s}\n".format(algo))

    start = datetime.now()
    # Train the model via KNN regression
    # k=5 for training
    kNN = KNeighborsRegressor(n_neighbors=5, weights='distance',algorithm=algo)
    kNN.fit(Input_train, Response_train)

    ## performance analysis parameters

    Error_KNN   = [None] * len(Input_test)
    predictedIV = Error_KNN

    ##predicted implied votatility
    predictedIV = kNN.predict(Input_test)
    Error_KNN   = abs(Response_test - predictedIV)


    # Model Evaluation

    print('The KNN Model Under Algorithm {:s} Peformance Summary as follows:'.format(algo.upper()))
    print('The MSE is           {:20.16f}'.format(get_MSE(Error_KNN)))
    print('The mean error is    {:20.16f}'.format(np.mean(Error_KNN)))
    print('The maximum error is {:20.16f}'.format(max(Error_KNN)))
    print('The minimum error is {:20.16f}'.format(min(Error_KNN)))
    print('Model running time:  {:20.16f} seconds'.format((datetime.now() - start).seconds))


print('\n')

# 'poly cab take more time than 'linear' and 'rbf'
#kernel_list=['linear', 'rbf']

# print Input_train

SVM_kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']

for kernel in SVM_kernel_list:
    print("\nSVM model is running under kernel: {:s}\n".format(kernel))

    start = datetime.now()
    t = svm.SVR(kernel=kernel, cache_size=500)
    t.fit(Input_train, Response_train)

    ##predicted implied votatility
    predictedSVMIV = t.predict(Input_test)


    Error_SVM = [None] * len(Input_test)
    Error_SVM   = abs(Response_test - predictedSVMIV)


# Model Evaluation

    print('The SVM Model Under Kernel {:s} Peformance Summary as follows:'.format(kernel.upper()))
    print('The MSE is           {:20.16f}'.format(get_MSE(Error_SVM)))
    print('The mean error is    {:20.16f}'.format(np.mean(Error_SVM)))
    print('The maximum error is {:20.16f}'.format(max(Error_SVM)))
    print('The minimum error is {:20.16f}'.format(min(Error_SVM)))
    print('Model running time:  {:20.16f} seconds'.format((datetime.now() - start).seconds))