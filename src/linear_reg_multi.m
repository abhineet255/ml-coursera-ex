%%%% Linear Regression with Multiple Variables/Features

%% Problem:
% The problem is to predict the price of a house for sale.
% Historic data for the price of houses is given along with two parameters:
%   size of the house (in square meters), and
%   number of bedrooms.
% We will use Gradient Descent to train a Linear Regression Model.
% We will also use Normal Equation to train the same model and compare results.
% Then use that model to predict the price of a 1650 sq-ft, 3 br house.

clear; close all; clc

function main()

    %% ================ Part 1: Feature Normalization ================

    % Since we have 2 variables, and their range is not similar, we will first normalize our data
    fprintf('Loading data ...\n');

    %% Load Data
    data = load('linear_reg_multi_d.txt');
    X = data(:, 1:2);
    y = data(:, 3);
    m = length(y);

    % Print out some data points
    fprintf('First 10 examples from the dataset: \n');
    fprintf(' x = [%.4f %.4f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

    % Scale features and set them to zero mean
    fprintf('Normalizing Features ...\n');

    % Compute the mean of all features in mu, and the range is sigma
    % Then normalize each feature as (x - mu) / sigma
    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    for i = 1:size(X,2)
        mu(i) = mean(X(:,i));
        sigma(i) = std(X(:,i));
        X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
    end

    X = X_norm;

    % Add intercept term to X
    X = [ones(m, 1) X];

    fprintf('First 10 examples from the dataset after normalization: \n');
    fprintf(' x = [%.0f %.4f %.4f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

    %% ================ Part 2: Gradient Descent ================

    fprintf('Running gradient descent ...\n');

    % Choose some alpha value
    alpha = [0.001, 0.01, 0.1, 1];
    num_iters = 1500;

    figure; hold on;
    plotColor = ['b', 'r', 'g', 'k']; 
    xlabel('Number of iterations');
    ylabel('Cost J');

    for i = 1:length(alpha)
        % Init Theta and Run Gradient Descent 
        theta = zeros(3, 1);
        [theta, J_history] = gradientDescentMulti(X, y, theta, alpha(i), num_iters);

        % Plot the convergence graph
        plot(1:numel(J_history), J_history, plotColor(i), 'LineWidth', 2);
    end

    legend('0.001', '0.01', '0.1', '1');
    hold off;

    % Now run the gradient descent with the chosen value of alpha
    alpha = 0.01;

    % Init Theta and Run Gradient Descent 
    theta = zeros(3, 1);
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

    % Display gradient descent's result
    fprintf('Theta computed from gradient descent: \n');
    fprintf(' %f \n', theta);
    fprintf('\n');

    % Estimate the price of a 1650 sq-ft, 3 br house
    % Remember that theta0 term should not be normalized
    XP = [1 1650 3];
    XP(:, 2:3) = (XP(:, 2:3) - mu) / sigma;
    price = XP * theta;

    fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'], price);

    %% ================ Part 3: Normal Equations ================

    fprintf('Solving with normal equations...\n');

    %% Load Data
    data = csvread('linear_reg_multi_d.txt');
    X = data(:, 1:2);
    y = data(:, 3);
    m = length(y);

    % Add intercept term to X
    X = [ones(m, 1) X];

    % Calculate the parameters from the normal equation
    theta = normalEqn(X, y);

    % Display normal equation's result
    fprintf('Theta computed from the normal equations: \n');
    fprintf(' %f \n', theta);
    fprintf('\n');


    % Estimate the price of a 1650 sq-ft, 3 br house
    % No need to normalize with normal equation method
    XP2 = [1 1650 3];
    price = XP2 * theta;
    fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price);
end

function J = computeCost(X, y, theta)
    %COMPUTECOST Compute cost for linear regression
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y

    % Number of training examples
    m = length(y);

    % term1 = h(x) = theta0 + theta1 * x1
    % term1 -> m x 1 matrix, containing the h(x) values for all m training examples
    term1 = X * theta;

    % term2 = h(x) - y
    % term2 -> m x 1 matrix, containing h(x) - y values for all m training examples
    term2 = term1 - y;

    % term3 = Summation of (h(x) - y) squared
    %   which is the product of each item in term2 matrix with itself
    %   which can be obtained by multiplying term2 with its transpose
    % term3 -> scalar value
    term3 = term2' * term2;

    % The cost is summation term divided by 2m
    J = term3 / (2 * m);
end

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iterations)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iterations) updates theta by 
    %   taking num_iterations gradient steps with learning rate alpha
    %   This code is exactly the same as the one used for single variable.

    % Number of training examples
    m = length(y);

    % History of cost function values over iterating theta values
    J_history = zeros(num_iterations, 1);

    for iter = 1:num_iterations

        % term1 = h(x) = theta0 + theta1 * x1
        % term1 -> m x 1 matrix, containing the h(x) values for all m training examples
        term1 = X * theta;

        % term2 = h(x) - y
        % term2 -> m x 1 matrix, containing h(x) - y values for all m training examples
        term2 = term1 - y;

        % term3 = (h(xi) - yi) . xi
        %   Each h(xi) - yi term must be multiplied by its corresponding xi
        %   Then all these values need to be summed over i=1:m
        %   This can be achieved by product of h(x) - y and x matrices
        % term3 -> 2 x 1 matrix, containing ((h(xi) - yi) . xi) for theta0 and theta1
        term3 = X' * term2;

        % Derivative term is term3 divided by m
        derivative = term3 / m;

        % Update theta matrix based on the derivative
        theta = theta - (alpha * derivative);

        % Save the cost J in every iteration
        % This records the value of cost function over iterating theta values
        J_history(iter) = computeCost(X, y, theta);
    end
end

function [theta] = normalEqn(X, y)
    %NORMALEQN Computes the closed-form solution to linear regression 
    %   NORMALEQN(X,y) computes the closed-form solution to linear 
    %   regression using the normal equations.

    theta = pinv(X' * X) * X' * y;
end

main();
