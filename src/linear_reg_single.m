%%%% Linear Regression with Single Variable/Feature

%% Problem:
% The problem is to predict the profits of a food truck in a city of given population.
% Historic data for the profits of the food truck in various cities and the population
%   of those cities is provided in linear_reg_single_d.txt
% We will use Gradient Descent to train a Linear Regression Model.
% Then use that model to predict the profits for cities of population 35,000 and 70,000.

clear; close all; clc

function main()
    % Linear Regression with Single Variable/Feature

    %% ======================= Part 1: Reading and Plotting Data =======================
    fprintf('Plotting Data ...\n')
    data = load('linear_reg_single_d.txt');

    % X is the first column of data, and y is the second column
    % X -> m x 1 matrix
    % y -> m x 1 matrix
    X = data(:, 1); y = data(:, 2);

    % Number of training examples
    m = length(y);

    % Open a new figure window
    figure;

    % Plot Data
    % 'rx' makes the data points appear as red crosses
    % MarkerSize 10 increases the size of each marker
    plot(X, y, 'rx', 'MarkerSize', 10);

    % Set the labels for the axes
    ylabel('Profit in $10,000s');
    xlabel('Population of City in 10,000s');

    %% ======================= Part 2: Cost Function =======================

    % Add a column of ones to X, representing x0
    X = [ones(m, 1), data(:,1)];

    % Initialize fitting parameters to 0 to begin with
    % theta -> 2 x 1 matrix => theta0 and theta1
    theta = zeros(2, 1);

    % Some gradient descent settings
    num_iterations = 1500;
    alpha = 0.01;

    fprintf('\nTesting the cost function ...\n')
    % compute and display initial cost
    J = computeCost(X, y, theta);
    fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
    fprintf('Expected cost value (approx) 32.07\n');

    % further testing of the cost function
    J = computeCost(X, y, [-1 ; 2]);
    fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
    fprintf('Expected cost value (approx) 54.24\n');

    %% ======================= Part 3: Gradient Descent =======================

    fprintf('\nRunning Gradient Descent ...\n')

    % Run gradient descent
    [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iterations);

    % Print theta to screen
    fprintf('Theta found by gradient descent:\n');
    fprintf('%f\n', theta);
    fprintf('Expected theta values (approx)\n');
    fprintf(' -3.6303\n  1.1664\n\n');

    % Plot the linear fit
    % Keep previous plot visible
    hold on;

    % Plotting h(x) values against the x values using dashes
    plot(X(:, 2), X * theta, '-')
    legend('Training data', 'Linear regression')

    % don't overlay any more plots on this figure
    hold off

    % Plot the values of cost function over iterations of gradient descent
    fprintf('\nPlotting Cost over gradient descent iterations ...\n')
    figure;
    plot(J_history)

    %% ======================= Part 4: Prediction =======================

    % Predict values for population sizes of 35,000 and 70,000
    predict1 = [1, 3.5] * theta;
    fprintf('For population = 35,000, we predict a profit of %f\n', predict1 * 10000);
    predict2 = [1, 7] * theta;
    fprintf('For population = 70,000, we predict a profit of %f\n', predict2 * 10000);

    %% ======================= Part 5: Visualizing J(theta_0, theta_1) =======================

    fprintf('Visualizing J(theta_0, theta_1) ...\n')

    % We will compute the cost function over a range of theta0 and theta1 values
    % Grid over which we will calculate J

    % theta0 values are spread from -10 to +10 with 100 values in between. Similarly theta1
    % theta0_vals, theta1_vals -> 100 x 1 matrices
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);

    % initialize J_vals to a matrix of 0s
    % J_vals -> 100 x 100 matrix
    J_vals = zeros(length(theta0_vals), length(theta1_vals));

    % Fill out J_vals
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = computeCost(X, y, t);
        end
    end

    % Because of the way meshgrids work in the surf command, we need to
    % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals';

    % Surface plot
    figure;
    surf(theta0_vals, theta1_vals, J_vals)
    xlabel('\theta_0'); ylabel('\theta_1');

    % Contour plot
    figure;

    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); ylabel('\theta_1');
    hold on;
    plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
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

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iterations)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iterations) updates theta by 
    %   taking num_iterations gradient steps with learning rate alpha

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

main();
