%%%% Multi-Class Logistic Regression Classifier

%% Problem:
% The problem is to predict the digit represented by a 20x20 grayscale image
%   of a hand written digit.
% We will use one-vs-all technique and train 10 logistic regression classifiers,
%   each for one output class.
% (The digit '0' will be represented by y = 10)

clear; close all; clc

function main()

    % 20x20 Input Images of Digits
    input_layer_size  = 400;

    % 10 labels, from 1 to 10
    num_labels = 10;    % (note that we have mapped "0" to label 10)

    %% =========== Part 1: Loading and Visualizing Data =============
    %  We start by first loading and visualizing the dataset.

    % Load Training Data
    fprintf('Loading and Visualizing Data ...\n')

    % The '.mat' file contains the matrices X and y, so we dont need to explicitly create them.
    load('handwritten_digits.mat');
    m = size(X, 1);

    % Randomly select 100 data points to display
    rand_indices = randperm(m);
    sel = X(rand_indices(1:100), :);

    displayData(sel);

    %% ============ Part 2a: Vectorize Logistic Regression ============

    % Test case for lrCostFunction
    fprintf('\nTesting lrCostFunction() with regularization');

    theta_t = [-2; -1; 1; 2];
    X_t = [ones(5,1) reshape(1:15,5,3)/10];
    y_t = ([1;0;1;0;1] >= 0.5);
    lambda_t = 3;
    [J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

    fprintf('\nCost: %f\n', J);
    fprintf('Expected cost: 2.534819\n');
    fprintf('Gradients:\n');
    fprintf(' %f \n', grad);
    fprintf('Expected gradients:\n');
    fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

    %% ============ Part 2b: One-vs-All Training ============
    fprintf('\nTraining One-vs-All Logistic Regression...\n')

    lambda = 0.1;
    [all_theta] = oneVsAll(X, y, num_labels, lambda);

    %% ================ Part 3: Predict for One-Vs-All ================

    pred = predictOneVsAll(all_theta, X);

    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
end

function [h, display_array] = displayData(X, example_width)
    %DISPLAYDATA Display 2D data in a nice grid
    %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    %   stored in X in a nice grid. It returns the figure handle h and the 
    %   displayed array if requested.

    % Set example_width automatically if not passed in
    if ~exist('example_width', 'var') || isempty(example_width) 
        example_width = round(sqrt(size(X, 2)));
    end

    % Gray Image
    colormap(gray);

    % Compute rows, cols
    [m n] = size(X);
    example_height = (n / example_width);

    % Compute number of items to display
    display_rows = floor(sqrt(m));
    display_cols = ceil(m / display_rows);

    % Between images padding
    pad = 1;

    % Setup blank display
    display_array = - ones(pad + display_rows * (example_height + pad), ...
                        pad + display_cols * (example_width + pad));

    % Copy each example into a patch on the display array
    curr_ex = 1;
    for j = 1:display_rows
        for i = 1:display_cols
            if curr_ex > m, 
                break; 
            end
            % Copy the patch

            % Get the max value of the patch
            max_val = max(abs(X(curr_ex, :)));
            display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                        pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                            reshape(X(curr_ex, :), example_height, example_width) / max_val;
            curr_ex = curr_ex + 1;
        end
        if curr_ex > m, 
            break; 
        end
    end

    % Display Image
    h = imagesc(display_array, [-1 1]);

    % Do not show axis
    axis image off

    drawnow;
end

function g = sigmoid(z)
    %SIGMOID Compute sigmoid functoon
    %   J = SIGMOID(z) computes the sigmoid of z.

    g = 1.0 ./ (1.0 + exp(-z));
end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization

    m = length(y); % number of training examples

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
    theta1 = theta;
    theta1(1) = 0;
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

    grad = grad(:);
end

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
    %ONEVSALL trains multiple logistic regression classifiers and returns all
    %the classifiers in a matrix all_theta, where the i-th row of all_theta 
    %corresponds to the classifier for label i

    m = size(X, 1);
    n = size(X, 2);

    all_theta = zeros(num_labels, n + 1);

    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    % Train multiple logistic regression classifiers and store their theta values to return
    for i = 1:num_labels
        initial_theta = zeros(n + 1, 1);
        options = optimset('GradObj', 'on', 'MaxIter', 50);
        [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), initial_theta, options);
        all_theta(i,:) = theta';
    end
end

function p = predictOneVsAll(all_theta, X)
    %PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    %are in the range 1..K, where K = size(all_theta, 1). 

    m = size(X, 1);
    num_labels = size(all_theta, 1);

    p = zeros(size(X, 1), 1);

    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    % For each classifier, compute the value of the hypothesis in h
    % Each row of h gives the probability of the input corresponding to each of K (1 to 10) 
    % Specifically, row i of h gives the K predicted values (each [0,1]) for i-th training example
    % z, h -> m x K matrices
    z = X * all_theta';
    h = sigmoid(z);

    % Find the maximum value and its index in the h vector
    % The label corresponding to the maximum value is the prediction
    [mx, mxi] = max(h, [], 2);
    p = mxi;
end

main();

