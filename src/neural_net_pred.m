%%%% Neural Network Classifier (without Parameter Optimization)

%% Problem:
% The problem is to predict the digit represented by a 20x20 grayscale image
%   of a hand written digit.
% We will use Neural Network with one hidden layer of (25 units).
% We are using pre-defined parameters (theta values) to just predict using the Neural Network.
% (The digit '0' will be represented by y = 10)

clear; close all; clc

function main()

    input_layer_size  = 400;  % 20x20 Input Images of Digits
    hidden_layer_size = 25;   % 25 hidden units
    num_labels = 10;          % 10 labels, from 1 to 10
                              % (note that we have mapped "0" to label 10)

    %% =========== Part 1: Loading and Visualizing Data =============

    % Load Training Data
    fprintf('Loading and Visualizing Data ...\n')

    load('handwritten_digits.mat');
    m = size(X, 1);

    % Randomly select 100 data points to display
    sel = randperm(size(X, 1));
    sel = sel(1:100);

    displayData(X(sel, :));
    pause;

    %% ================ Part 2: Loading Pameters ================
    % Load some pre-initialized neural network parameters.

    fprintf('\nLoading Saved Neural Network Parameters ...\n')

    % Load the weights into variables Theta1 and Theta2
    load('neural_net_pred.mat');

    %% ================= Part 3: Implement Predict =================
    %  After training the neural network, we would like to use it to predict
    %  the labels.

    pred = predict(Theta1, Theta2, X);

    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

    %  We will now run through some random samples one at a time to see their prediction.

    %  Randomly permute examples
    rp = randperm(m);

    for i = 1:4
        % Display 
        fprintf('\nDisplaying Example Image\n');
        displayData(X(rp(i), :));

        pred = predict(Theta1, Theta2, X(rp(i),:));
        fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

        pause;
    end
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

function p = predict(Theta1, Theta2, X)
    %PREDICT Predict the label of an input given a trained neural network
    %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    %   trained weights of a neural network (Theta1, Theta2)

    % Useful values
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    % Implement the Feedforward Propagation Algorithm to compute the output

    % Add x0, and set that as A1
    A1 = [ones(m, 1) X];

    % Compute Z2 from A1, then A2 from Z2, and then add the bias term to A2
    Z2 = A1 * Theta1';
    A2 = sigmoid(Z2);
    A2 = [ones(m, 1) A2];

    % Similarly for the output layer. Note that we don't need to add bias term to the output layer
    Z3 = A2 * Theta2';
    A3 = sigmoid(Z3);

    % From the output values, find the maximum amongst all the classes. That is out prediction
    [mx, mxi] = max(A3, [], 2);
    p = mxi;
end

main();
