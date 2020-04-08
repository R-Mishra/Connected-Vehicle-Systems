## IMPLEMENTING THE ENTIRE MODEL PIPELINE

## Prev. Update: March 30, 2020.
## Last Updated: March 31, 2020.
## Added an input for multiplier and learning rate


## Author: Rohit Mishra

## _____________________________________________________________________________________________________________________________________________

## This file contains the following method :-
## main_function() - It combines all the functions in this package in such a way that it takes 

##  1. Driving data
##  2. Collision outcome data
##  3. Data split vector (Number of instances per data point)
##  4. Scenario vector - Information regarding the scenario and the warning parameters
##  5. Multiplier - The number of time the data has to be multiplied
##  6. k_folds - Number of folds for cross-validation
##  7. No. of batches 
##  8. Epochs

## and returns the confusion matrix and test predictions 

## ______________________________________________________________________________________________________________________________________________

from cvs_package.data_prep import data_multiplier_and_shaper, test_train_split 
from cvs_package.mdl_train import model_init
from cvs_package.mdl_test import test

def main_function(driving_data, output_data,data_split_vector, scenario_vector, k_folds = 8,no_of_batches = 4, epoch = 20):

    print("How many times over should the data be multiplied?")
    multiplier = int(input())
    input_data, output_data, data_split_vector, scenario_vector = data_multiplier_and_shaper(driving_data, output_data,data_split_vector, scenario_vector,multiplier)
    print("There are {} inputs to the LSTM. How many folds do you want for your cross-validation, default being 8?".format(input_data.shape[0]))
    k_folds = input()
    k_folds = int(k_folds)
    print("The number of folds: {}".format(k_folds))
    print("What is the number of batches you desire the training data be split into? Default is set at 4")
    no_of_batches = input()
    no_of_batches = int(no_of_batches)
    print("The number of batches: {}".format(no_of_batches))
    print("What is the number of epochs for which the training should be carried out? Default is set at 20")
    epoch = input()
    epoch = int(epoch)
    print("The number of epoch: {}".format(epoch))
    print("What is the learning rate desired?")
    learning_rate_usr = float(input())
    print("The hyperparameters fixed for this experiemntal run are:\n \
        Multiplier    = {} \n \
        Folds         = {} \n \
        Batches       = {} \n \
        Epochs        = {} \n \
        Learning Rate = {} \n ".format(multiplier,k_folds,no_of_batches,epoch,learning_rate_usr))

   
    true_positive  = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    
    for k in range(k_folds):
        input_train_data, input_train_target, input_test_data, test_target,train_events, \
        train_datasplit, test_datasplit, train_scenario, test_scenario = test_train_split(k_folds, input_data, output_data, data_split_vector, scenario_vector, k, multiplier)
        collision_predictor = model_init(input_train_data, input_train_target, no_of_batches, data_split_vector, train_datasplit, train_scenario, epoch, learning_rate_usr)
        tp, fp, fn, tn, test_predictions = test(input_test_data, test_target, collision_predictor, test_datasplit, test_scenario)
        true_positive  += tp
        false_positive += fp
        false_negative += fn
        true_negative  += tn
        print("accuracy for fold {} is {}".format(k,((tp+tn)/(tp+fp+fn+tn))))
        print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp,tn,fp,fn))

    
    return true_positive,true_negative,false_positive, false_negative, test_predictions
                     