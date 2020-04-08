# Connected-Vehicle-Systems

A connected vehicle paradigm involves communication among the vehicles and the roadside infrastructure. This allows for transmission of crucial information which has the potential to make driving a safer experience. These vehicles make use of advanced sensor technology and the internet for information exchange. Below, is an image showing a connected vehicle. It gives an idea of how the technological components come together to make a connected vehicle possible.

![A connected Vehicle](https://www.its.dot.gov/cv_basics/images/cv_basics_car_viewLarger.png)

Figure 1. A connected vehicle with its components that allow it to communicate with the connected vehicle infrastructure

The advances in the connected vehicle systems (CVS) support the wireless communication between vehicles. The data delivered via this type of technology includes positional and movement information of surrounding connected vehicles. The vehicle-to-vehicle communications enables a vehicle to detect and evaluate the potential hazard on the road, generate warnings before potentially life-threatening scenarios take place, and assist the driver in taking preventive actions in advance. 

Empirical studies suggest that warning parameters of connected vehicles, such as warning timing and warning style, are crucial in determining the driver response and the consequent safety outcome. However, few mathematical models have been developed to predict the collision rates in connected vehicles by quantifying the impact of warning parameters. In a previous work of ours, simple models such as Support Vector Machine and Logistic Regression have been used to predict the collision rates in hazard events with a prediction accuracy of about 70%. The previous models were established based on aggregated driving data which is only available on the aftermath of the warning. In order to select the optimal levels of the warning parameters in a real time basis before a warning is generated, it is necessary to model the impacts of warning parameters on the collision rates and utilize the time-series feature of the driving data. This motivated us to work towards developing more complex neural networks-based models using time-series data of human drivers collected from driving simulator experiments to predict the safety outcomes. 

In this work, we analyzed the time-series data of sixty four drivers’ driving performance before and after warnings along with data from the surrounding vehicles. We developed the Long Short-Term Memory Neural Network models to predict the collision rates by considering various warning parameters and different hazard scenario features. We also develop some convoluted neural network models to capture the trends in the data and predcit the outcome of hazardous events. 


