%%%% Regularized Logistic Regression with Order-6 Feature Mapping

%% Problem:
% To predict whether a microchip passes Quality Assurance, given their scores on two tests.
% As can be seen from the data, linear fitting model will not work in this case.
% A sixth-order polynomial is used for feature mapping.
% Then Regularization is used to arrive at an optimal model.
% We also see the problem of overfitting and underfitting, and how it is resolved by Regularization.

clear ; close all; clc

function main()

    %% Load Data
    %  The first two columns contains the X values and the third column
    %  contains the label (y).

    data = load('log_reg_regularized.txt');
    X = data(:, [1, 2]); y = data(:, 3);

    % Create New Figure
    figure; hold on;
    plotData(X, y);

    % Put some labels
    hold on;

    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')

    % Specified in plot order
    legend('y = 1', 'y = 0')
    hold off;


    %% =============== Part 1: Regularized Logistic Regression ===============

    % Add Polynomial Features
    % There are 2 features x1 and x2. All polynomial terms upto 6th degree are added
    % x1, x2, x1 ^ 2, x2 ^ 2, x1 . x2, x1 ^ 2 . x2    ...   x1 ^ 5 . x2, x1 ^ 6, x2 ^ 6
    X = mapFeature(X(:, 1), X(:, 2));

    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);

    % Set regularization parameter lambda to 1
    lambda = 0.1;

    % Compute and display initial cost and gradient for regularized logistic regression
    [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

    fprintf('Cost at initial theta (zeros): %f\n', cost);
    fprintf('Expected cost (approx): 0.693\n');
    fprintf('Gradient at initial theta (zeros) - first five values only:\n');
    fprintf(' %f \n', grad(1:5));
    fprintf('Expected gradients (approx) - first five values only:\n');
    fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

    % Compute and display cost and gradient with all-ones theta and lambda = 10
    test_theta = ones(size(X,2),1);
    [cost, grad] = costFunctionReg(test_theta, X, y, 10);

    fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
    fprintf('Expected cost (approx): 3.16\n');
    fprintf('Gradient at test theta - first five values only:\n');
    fprintf(' %f \n', grad(1:5));
    fprintf('Expected gradients (approx) - first five values only:\n');
    fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

    %% ============= Part 2: Regularization and Accuracies =============
    % Here we will try various values of lambda to see how that affects the cost function.

    fprintf('\nComparing results with different values of lambda...\n');

    % Create New Figure
    figure; hold on;

    % Plot Data
    plotData(X(:,2:3), y);
    hold on

    % Set regularization parameter lambda to 1
    lambda = [0 1 10 100];
    lineColor = ['b' 'r' 'g' 'k'];

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    for i = 1:length(lambda)
        % Initialize fitting parameters
        initial_theta = zeros(size(X, 2), 1);

        [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda(i))), initial_theta, options);

        % Plot Boundary
        plotDecisionBoundary(theta, X, y, lineColor(i));
    end

    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')

    legend('y = 1', 'y = 0')
    hold off;

    % Compute accuracy on our training set
    p = predict(theta, X);

    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
end

function features = mapFeature(X1, X2)
    % MAPFEATURE Feature mapping function to polynomial features
    %   MAPFEATURE(X1, X2) maps the two input features
    %   to quadratic features used in the regularization exercise.
    %
    %   Returns a new feature array with more features, comprising of 
    %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    %
    %   Inputs X1, X2 must be the same size
    %

    degree = 6;

    % Create features as m x 1 matrix, filled with 1's
    features = ones(size(X1(:, 1)));

    for i = 1:degree
        for j = 0:i
            % Append a column to features, representing one polynomial term for all training data
            features(:, end + 1) = (X1 .^ (i - j)) .* (X2 .^ j);
        end
    end
end

function plotData(X, y)
    %PLOTDATA Plots the data points X and y into a new figure 
    %   PLOTDATA(x,y) plots the data points with + for the positive examples
    %   and o for the negative examples. X is assumed to be a Mx2 matrix.

    % Find Indices of Positive and Negative Examples
    % pos will have the indices of elements with y = 1, similarly for neg
    pos = find(y == 1);
    neg = find(y == 0);

    % Plot Examples, use + for y=1, and o for y=0
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

    hold off;
end

function plotDecisionBoundary(theta, X, y, lineColor)
    % PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    % the decision boundary defined by theta
    %   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    %   positive examples and o for the negative examples. X is assumed to be
    %   MxN matrix, N >= 3, where the first column is an all-ones column for the intercept.

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
        % Evaluate z = theta * x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j)) * theta;
            end
        end
        z = z'; % important to transpose z before calling contour

        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2, 'LineColor', lineColor)
    end
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

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters.

    % Number of training examples
    m = length(y);

    % h = g(z), where g = sigmoid function, z = X * theta
    % h -> m x 1 matrix
    h = sigmoid(X * theta);

    % logh, logh1 -> m x 1 matrices
    logh = log(h);
    logh1 = log(1 - h);

    % term1 is the usual summation term, without the regularization component
    term1 = ((y' * logh) + ((1 - y)' * logh1)) * -1 / m;

    % term2 is the regularization component of the cost function
    % Note that theta0 needs to be excluded from the regularization
    theta1 = theta(2:end, :);
    term2 = (theta1' * theta1) * lambda / (2 * m);

    J = term1 + term2;

    % Compute the gradient terms below

    % gterm1 is the summation of (h(xi) - yi) . xi
    gterm1 = ((h - y)' * X / m)';

    % gterm2 is the regularization part of the derivative
    % Note that we set it to 0 from theta0
    gterm2 = theta * lambda / m;
    gterm2(1) = 0;

    grad = gterm1 + gterm2;
end

main();
