## INITIALIZATION AND TRAINING OF LSTM MODEL

## Prev. Update: March 30, 2020.
## Last Updated: March 31, 2020.
## Added learining_rate_usr

## Author: Rohit Mishra

## _____________________________________________________________________________________________________________________________________________

## This file contains the following methods :-
## model_init() - initalize te LSTM model and call the training function
## model_trainer() - trains the model on the dataset and returns the trained model

## ______________________________________________________________________________________________________________________________________________

from cvs_package.lstm_model import collisionClassifier
import torch 
import torch.nn as nn

def model_init(input_train_data, input_train_target, no_of_batches, data_split_vector, train_datasplit,train_scenario, epoch, learning_rate_usr):
    
    collision_predictor = collisionClassifier(no_of_features = input_train_data.size()[2],features_to_hidden = 20, \
                                          features_in_hidden = 5 , max_sequence_length = max(data_split_vector))
    
    batch_size = int(input_train_data.size()[0]/no_of_batches)
    optimizer = torch.optim.Adam(collision_predictor.parameters(), lr=learning_rate_usr)
    optimizer2 = torch.optim.SGD(collision_predictor.parameters(), lr=0.01)

    return model_trainer(input_train_data, input_train_target, no_of_batches, data_split_vector,\
                         train_datasplit, train_scenario, batch_size, optimizer, optimizer2, collision_predictor, epoch)

def avg_error(linear_output, target):
    weights = torch.tensor([1.,2.3]) # Use smaller value 
    #loss = nn.CrossEntropyLoss(weight = weights)
    #loss = nn.BCELoss(reduction = 'none')
    loss = nn.NLLLoss(weight = weights)
    error = loss(linear_output,target)
    return error

def model_trainer(input_train_data, input_train_target, no_of_batches, data_split_vector,\
                         train_datasplit, train_scenario, batch_size, optimizer, optimizer2, collision_predictor, epoch):
    
    for i in range(epoch):

        for j in range(no_of_batches):

            pred_tensor = torch.zeros(batch_size,2)
            target = input_train_target
            target_tensor = target[j*batch_size:(j+1)*batch_size] 
            
            for k in range (batch_size):

                event_no = (batch_size*j)+k
                prediction = collision_predictor.forward_pass(input_train_data[event_no,:,:].unsqueeze(0),\
                                                                      train_datasplit[event_no, 0], train_scenario[event_no,:])
                pred_tensor[k] = prediction
        
          
            error = avg_error(pred_tensor,target_tensor)
            collision_predictor.zero_grad()
            if i <= 10:
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
            else:
                optimizer2.zero_grad()
                error.backward()
                optimizer2.step()
        
        print("The error for epoch {} was: {}".format((i+1),error))
            
    return collision_predictor

