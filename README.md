# Machine Learning Algorithms
The implementation of various Machine Learning algorithms, as taught in the Coursera course by Andrew Ng.

This repo consists of implementation of the ML algorithms in Octave.

## Linear Regression with Single Variable/Feature
Predict the success of a food truck.

Given the data for profits of the food truck in various cities of given population.

Gradient Descent algorithm is used to fit the given data.

[Code](src/linear_reg_single.m)

[Output](output/linear_reg_single.md)


## Linear Regression with Multiple Variables/Features
Predict the price of a house for sale.

Training Set is the size of house (in square feet) and the number of bedrooms, along with the price of the house.

This problem is solved by both Gradient Descent and Normal Equation algorithms.

[Code](src/linear_reg_multi.m)

[Output](output/linear_reg_multi.md)


## Logistic Regression
Predict whether an application will get admission into university based on the score on two tests.

Training Set is historical data of admission based on the score of the two tests.

Logistic Regression model is trained using Gradient Descent first, then using advanced methods. Octave's fminunc() method is used to optimally arrive at the theta values which minimize the cost function.

[Code](src/log_reg.m)

[Output](output/log_reg.md)


## Disclaimer
P.S.: I do not claim credit for creating the problems and their solutions. These are the exercises provided as part of the Machine Learn Course. I have compiled them in a consumable form for my own learning.
