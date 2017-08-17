%%%% Logistic Regression

%% Problem:
% The problem is to predict whether a student gets admitted into a university.
% Historic data for the previous applicants is given along with their scores
%   on two subjects and the decision of admission.
% We will build a classification model that will estimate the probability of an applicant's 
%   probability of admission based on the scores of those two exams.
% We will use Gradient Descent to train a Logistic Regression Model.
% Then we will use that model to predict the admission decision of a student with
%   a score of 45 in exam 1 and a score of 85 in exam 2.

clear; close all; clc

function main()

    %% Load Data
    %  The first two columns contains the exam scores and the third column
    %  contains the label.

    data = load('log_reg.txt');
    X = data(:, [1, 2]); y = data(:, 3);

    %% ==================== Part 1: Plotting ====================
    %  We start the exercise by first plotting the data to understand the 
    %  the problem we are working with.

    fprintf(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);

    plotData(X, y);

    % Labels and Legend
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')

    % Specified in plot order
    legend('Admitted', 'Not admitted')
    hold off;

    %% ==================== Part 2: Compute Cost and Gradient ====================

    %  Setup the data matrix appropriately, and add ones for the intercept term
    [m, n] = size(X);

    % Add intercept term to x
    X = [ones(m, 1) X];

    % Initialize fitting parameters to 0
    initial_theta = zeros(n + 1, 1);

    % Compute and display initial cost and gradient
    [cost, grad] = costFunction(initial_theta, X, y);

    fprintf('Cost at initial theta (zeros): %f\n', cost);
    fprintf('Expected cost (approx): 0.693\n');
    fprintf('Gradient at initial theta (zeros): \n');
    fprintf(' %f \n', grad);
    fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

    % Compute and display cost and gradient with non-zero theta
    test_theta = [-24; 0.2; 0.2];
    [cost, grad] = costFunction(test_theta, X, y);

    fprintf('\nCost at test theta: %f\n', cost);
    fprintf('Expected cost (approx): 0.218\n');
    fprintf('Gradient at test theta: \n');
    fprintf(' %f \n', grad);
    fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

    %% ==================== Part 3: Optimizing using fminunc  ====================
    % Use a built-in function (fminunc) to find the optimal parameters theta,
    %   without gradient descent.

    % Set options for fminunc
    % Set the GradObj option to on, which tells fminunc that our function
    %   returns both the cost and the gradient. This allows fminunc to use
    %   the gradient when minimizing the function.
    % Set the MaxIter option to 400, so that fminunc will run for at most 400 steps
    %   before it terminates.
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Run fminunc to obtain the optimal theta
    % This function will return theta and the cost
    % To specify the actual function we are minimizing, we use a short-hand
    %   for specifying functions with the @(t) ( costFunction(t, X, y) ).
    %   This creates a function, with argument t, which calls your costFunction.
    %   This allows us to wrap the costFunction for use with fminunc.
    [theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

    % Print theta to screen
    fprintf('Cost at theta found by fminunc: %f\n', cost);
    fprintf('Expected cost (approx): 0.203\n');
    fprintf('theta: \n');
    fprintf(' %f \n', theta);
    fprintf('Expected theta (approx):\n');
    fprintf(' -25.161\n 0.206\n 0.201\n');

    % Plot Boundary
    plotDecisionBoundary(theta, X, y);

    % Put some labels 
    hold on;
    % Labels and Legend
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')

    % Specified in plot order
    legend('Admitted', 'Not admitted')
    hold off;

    %% ============== Part 4: Predict and Accuracies ==============
    %  After learning the parameters, you'll like to use it to predict the outcomes
    %  on unseen data. In this part, you will use the logistic regression model
    %  to predict the probability that a student with score 45 on exam 1 and 
    %  score 85 on exam 2 will be admitted.
    %
    %  Furthermore, you will compute the training and test set accuracies of 
    %  our model.
    %
    %  Your task is to complete the code in predict.m

    %  Predict probability for a student with score 45 on exam 1 
    %  and score 85 on exam 2 

    prob = sigmoid([1 45 85] * theta);
    fprintf(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob);
    fprintf('Expected value: 0.775 +/- 0.002\n\n');

    % Compute accuracy on our training set
    p = predict(theta, X);

    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    fprintf('Expected accuracy (approx): 89.0\n');
    fprintf('\n');
end

function g = sigmoid(z)
    %SIGMOID Compute sigmoid function
    %   g = SIGMOID(z) computes the sigmoid of z.

    % e is the exponent term = e ^ (-z)
    e = exp(-1 * z);

    % g = 1 / (1 + e)
    % Since z and e can be matrices, we use ./ operator
    g = 1 ./ (1 + e);
end

function [J, grad] = costFunction(theta, X, y)
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.

    % Number of training examples
    m = length(y);

    % h = g(z), where g = sigmoid function, z = (theta0 + theta1 * x)
    % h -> m x 1 matrix
    h = sigmoid(X * theta);

    % logh, logh1 -> m x 1 matrices
    logh = log(h);
    logh1 = log(1 - h);

    % term1 is the summation term
    % Multiplying matrices like below multiplies individual matrix terms and then sums them up
    % term1, J -> scalar values
    term1 = (y' * logh) + ((1 - y)' * logh1);
    J = term1 * -1 / m;

    % Compute the gradient terms below

    % gterm1 is h(x) - y
    % gterm1 -> m x 1 matrix
    gterm1 = h - y;

    % gterm2 is the summation term
    % gterm2 = summation of (h(xi) - yi) . xi
    % Multiplying matrices like below does the above scalar multiplication,
    %   then sums up all the terms over i = 1 to m
    % gterm2 -> n+1 x 1 matrix
    gterm2 = X' * gterm1;

    % grad -> n+1 x 1 matrix
    grad = gterm2 / m;
end

function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Create New Figure
    figure; hold on;

    % Find Indices of Positive and Negative Examples
    % pos will have the indices of elements with y = 1, similarly for neg
    pos = find(y == 1);
    neg = find(y == 0);

    % Plot Examples, use + for y=1, and o for y=0
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

    hold off;
end

function plotDecisionBoundary(theta, X, y)
    % PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    % the decision boundary defined by theta
    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    %   positive examples and o for the negative examples. X is assumed to be
    %   MxN matrix, N >= 3, where the first column is an all-ones column for the intercept.

    % Plot Data
    plotData(X(:,2:3), y);
    hold on

    if size(X, 2) <= 3
        % Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X(:, 2)) - 2,  max(X(:, 2)) + 2];

        % Calculate the decision boundary line
        plot_y = (-1 ./ theta(3)) .* (theta(2) .* plot_x + theta(1));

        % Plot, and adjust axes for better viewing
        plot(plot_x, plot_y)
        
        % Legend, specific for the exercise
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
    else
        % Here is the grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);

        z = zeros(length(u), length(v));
        % Evaluate z = theta*x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j))*theta;
            end
        end
        z = z'; % important to transpose z before calling contour

        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off
end

function p = predict(theta, X)
    %PREDICT Predict whether the label is 0 or 1 using learned logistic 
    %regression parameters theta

    % Number of training examples
    m = size(X, 1);

    h = sigmoid(X * theta);
    o = find(h >= 0.5);

    p = zeros(m, 1);
    p(o) = 1;
end

main();
