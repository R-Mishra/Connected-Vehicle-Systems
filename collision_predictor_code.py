import numpy as np
import pandas as pd
import cvs_package

print("Please enter the path to the directory in which the data is stored")
path = input()

# importing the time series input data

filepath_main_data = path+"cleaned_data_version_01.xlsx"
driving_data = pd.read_excel(filepath_main_data)
driving_data = driving_data.drop(['padek2','padel2','trans_gear', 'Unnamed: 0'],axis = 1)
#driving_data = driving_data.values


# importing the collision outcome for the corresponding input data

filepath_output = path+"output_data_version_01.xlsx"
output_data = pd.read_excel(filepath_output)
output_data = output_data.drop(['Unnamed: 0'], axis = 1)
output_data = output_data.values


# importing the data split vector which stores the length of each time series data

filepath_data_split = path+"/data_split_version_01.xlsx"
data_split_vector = pd.read_excel(filepath_data_split)
data_split_vector = data_split_vector.drop(['Unnamed: 0'], axis = 1)
data_split_vector = data_split_vector.values


# importing the scenario data 

filepath_scenario_data = path+"/scenario_data_version_01.xlsx"
scenario_vector = pd.read_excel(filepath_scenario_data)
scenario_vector = scenario_vector.drop(['Unnamed: 0'], axis = 1)
scenario_vector = scenario_vector.values


print("The shape of the input driving data is:{}\
The shape of the output data is: {}\
The shape of the data split vector is: {}\
The shape of the scenario vector is: {}".format(driving_data.shape, output_data.shape, data_split_vector.shape, scenario_vector.shape))

true_positive,true_negative,false_positive, false_negative, test_predictions = cvs_package.base.main_function(driving_data, 
	output_data,data_split_vector, scenario_vector, k_folds = 8,no_of_batches = 4, epoch = 20)

sum = (true_positive + true_negative + false_positive + false_negative)/100
print("true positive: {} \
true negative: {} \
false positive: {} \
false negative: {}".format(true_positive/sum,true_negative/sum,false_positive/sum, false_negative/sum))

print("accuracy: {}".format((true_positive + true_negative)/(true_positive + true_negative+ false_positive + false_negative)))
