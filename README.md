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


## Logistic Regression Classifier
Predict whether an applicant will get admission into university based on the score on two tests.

Training Set is historical data of admission based on the score of the two tests.

Logistic Regression model is trained using Gradient Descent first, then using advanced methods. Octave's fminunc() method is used to optimally arrive at the theta values which minimize the cost function.

[Code](src/log_reg.m)

[Output](output/log_reg.md)


## Regularized Logistic Regression with Order-6 Feature Mapping
Predict whether a chip will pass QA based on the score on two tests.

Training Set is historical data of QA based on the score of the two tests. The data is complicated so cannot be modeled using a linear model. An order-6 Logistic Regression model is trained to fit the data.

[Code](src/log_reg_regularized.m)

[Output](output/log_reg_regularized.md)


## Multi-class Logistic Regression Classifier
Given images of hand-written digits, recognize the digit.

Training Set is a 20 pixel by 20 pixel grayscale image of the digit, so 400 pixel values are provided as floating point values representing the grayscale intensity at that location.

We will use a 10-class logistic regression classifier to predict the digit.

[Code](src/log_reg_multi.m)

[Output](output/log_reg_multi.md)


## Disclaimer
P.S.: I do not claim credit for creating the problems and their solutions. These are the exercises provided as part of the Machine Learn Course. I have compiled them in a consumable form for my own learning.

P.P.S.: I could have modularized the code more by writing functions in their dedicated files and then re-using them. But I prefer to see them as one file for faster reference while learning.
