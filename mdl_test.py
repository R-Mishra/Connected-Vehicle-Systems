## TESTING LSTM MODEL

## Last Updated: March 30, 2020.

## Author: Rohit Mishra

## _____________________________________________________________________________________________________________________________________________

## This file contains the following methods :-
## test() - takes as its input the trained model and the testing data and returns the confusion matrix along with the model predictions

## ______________________________________________________________________________________________________________________________________________

import numpy as np

def test(input_test_data, test_target, collision_predictor, test_datasplit, test_scenario):
    
    test_target = np.array(test_target)
    test_predictions = np.zeros(test_target.shape)
    event_no = 0
    for i in range(0,input_test_data.shape[0]):
        test_output = collision_predictor.forward_pass(input_test_data[i,:,:].unsqueeze(0), \
                                                                test_datasplit[event_no,0], test_scenario[event_no,:])
        if test_output[0] < test_output[1]:
            test_predictions[i] = 1
        else :
            test_predictions[i] = 0
        event_no += 1
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(0, input_test_data.shape[0]):
        if ((test_predictions[i] == test_target[i]) & (test_predictions[i] == 1)):
            tp += 1
        if ((test_predictions[i] == test_target[i]) & (test_predictions[i] == 0)):
            tn += 1
        elif((test_predictions[i] != test_target[i]) & (test_predictions[i] == 1)):
            fp += 1
        elif((test_predictions[i] != test_target[i]) & (test_predictions[i] == 0)):
            fn += 1
    return tp, fp, fn, tn, test_predictions