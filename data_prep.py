## DATA PREPARATION FOR LSTM MODEL ARCHITECTURE 

## Prev. Update: March 30, 2020.
## Last Updated: March 31, 2020.
## Changed the padding method so as to pre-pad the input data; 

## Author: Rohit Mishra

## _____________________________________________________________________________________________________________________________________________

## This file contains the following methods :-
## data_multiplier_and_shaper() - This method is linked to data_scaler() and padding_method()
## and returns data multiplied  
## data_scaler()
## padding_method()
## test_train_split()

## ______________________________________________________________________________________________________________________________________________

import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import torch


def data_multiplier_and_shaper(driving_data, output_data, data_split_vector, scenario_vector, multiplier):
    
    driving_data_large = driving_data
    output_data_large = output_data
    data_split_vector_large = data_split_vector
    scenario_vector_large = scenario_vector
    for i in range (0,multiplier-1):
        driving_data_large = np.append(driving_data_large, driving_data)
        output_data_large = np.append(output_data_large,output_data)
        data_split_vector_large = np.append(data_split_vector_large, data_split_vector)
        scenario_vector_large = np.append(scenario_vector_large, scenario_vector)
    x = int(driving_data_large.shape[0]/12)
    
    driving_data_large = driving_data_large.reshape(x,12)
    output_data_large = output_data_large.reshape(output_data_large.shape[0],1)
    data_split_vector_large = data_split_vector_large.reshape(data_split_vector_large.shape[0],1)
    scenario_vector_large = scenario_vector_large.reshape(int(scenario_vector_large.shape[0]/4),4)
    
    print(driving_data_large.shape, data_split_vector_large.shape,scenario_vector_large.shape, output_data_large.shape)
    a = len(data_split_vector)
    b = len(data_split_vector_large)
    for i in range(a,b):
        driving_data_large[a,:] = driving_data_large[a,:] + \
        np.random.normal(loc = 0, scale = 1, size = (1,driving_data.shape[1]))

    return data_scaler(driving_data_large, data_split_vector_large) , output_data_large, data_split_vector_large, scenario_vector_large


def data_scaler(driving_data, data_split_vector):
    
    #print(driving_data.loc[driving_data.index.isna()])
    no_of_data_points = driving_data.shape[0]
    no_of_columns = driving_data.shape[1]

    scaler = MinMaxScaler(feature_range = (0,1))
    scaledData = scaler.fit_transform(driving_data)
    scaledData = scaledData.reshape([1, no_of_data_points, no_of_columns])
    
    return padding_method(scaledData, data_split_vector)


def padding_method(scenario_data, data_split_vector):

    scenario_tensor = np.zeros((len(data_split_vector),max(max(data_split_vector)),scenario_data.shape[2]))

    l = scenario_tensor.shape[1]
    j = 0    
    for idx, scen_length in enumerate(data_split_vector):
        for i in range (max(scen_length)):
            scenario_tensor[idx,i+l-scen_length,:] = (scenario_data[0,j+i,:])
        j += max(scen_length)
        
    print("The input data shape is {}".format(scenario_tensor.shape))
    
    return scenario_tensor


def test_train_split(k_folds, input_data, output_data, data_split_vector, scenario_vector, k, multiplier): # k is the fold number
    
    independent_events = input_data.shape[0]
    data_per_fold = int(independent_events/k_folds)
    fold_size = int(input_data.shape[0]/(multiplier*k_folds)) # fold size per replica
    train_events = (k_folds - 1)*data_per_fold
    input_test_data = []
    input_train_data = []
    test_target = []
    input_train_target =[]
    train_datasplit = []
    test_datasplit = []
    train_scenario = []
    test_scenario = []
    
    for l in range(k_folds):
        if l == k:
            for z in range(multiplier):
                delta = int(z*input_data.shape[0]/multiplier)+ int(l*fold_size)
                input_test_data = np.append(input_test_data, input_data[delta:delta+fold_size,:,:])
                test_target = np.append(test_target, output_data[delta:delta+fold_size,0])
                test_datasplit = np.append(test_datasplit, data_split_vector[delta:delta+fold_size,0]) 
                test_scenario = np.append(test_scenario, scenario_vector[delta:delta+fold_size,:])
        else:
            for z in range(multiplier):
                delta = int(z*input_data.shape[0]/multiplier)+ int(l*fold_size)
                input_train_data = np.append(input_train_data, input_data[delta:delta+fold_size,:,:])
                input_train_target = np.append(input_train_target, output_data[delta:delta+fold_size,0])
                train_datasplit = np.append(train_datasplit, data_split_vector[delta:delta+fold_size,0]) 
                train_scenario = np.append(train_scenario, scenario_vector[delta:delta+fold_size,:])
    
    input_test_data = torch.FloatTensor(input_test_data.reshape(test_datasplit.shape[0],input_data.shape[1],input_data.shape[2]))
    input_train_data = torch.FloatTensor(input_train_data.reshape(train_datasplit.shape[0],input_data.shape[1],input_data.shape[2]))
    test_target = torch.LongTensor(test_target.reshape(test_target.shape[0]))
    input_train_target =torch.LongTensor(input_train_target.reshape(input_train_target.shape[0]))
    train_datasplit = train_datasplit.reshape(train_datasplit.shape[0],1)
    test_datasplit = test_datasplit.reshape(test_datasplit.shape[0],1)
    train_scenario = torch.FloatTensor(train_scenario.reshape(train_datasplit.shape[0],4))
    test_scenario = torch.FloatTensor(test_scenario.reshape(test_datasplit.shape[0],4))

    return input_train_data, input_train_target, input_test_data, test_target, train_events, train_datasplit, test_datasplit, train_scenario, test_scenario