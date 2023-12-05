clc
clear

% Define the neural network architecture
inputNodes = 10;
hiddenNodes = 5;
outputNodes = 1;
numSamples = 100;

% Initialize the network's weights
W1 = randn(hiddenNodes, inputNodes);
W2 = randn(outputNodes, hiddenNodes);
b1 = randn(hiddenNodes, 1);
b2 = randn(outputNodes, 1);

% Define the activation function (sigmoid)
sigmoid = @(x) 1./(1 + exp(-x));

% Define the training data
X = randn(numSamples, inputNodes);
Y = randn(numSamples, outputNodes);

% Set the learning rate and number of epochs
learningRate = 0.2;
maxEpochs = 250;

% Initialize the Jacobian and Hessian matrices
J = zeros(hiddenNodes, inputNodes);
% Hessian = zeros(hiddenNodes*outputNodes + outputNodes*hiddenNodes, hiddenNodes*outputNodes + outputNodes*hiddenNodes);
Hessian = zeros(10);

% Initialize the root mean squared error (NMSE)
NMSE = zeros(maxEpochs, 1);

for epoch = 1:maxEpochs
    % Initialize the epoch's NMSE
    epochNMSE = 0;
    epochError = 0;
    
    % Iterate through each training pattern
    for i = 1:size(X, 1)
        
        % Forward pass
        a1 = sigmoid(W1*X(i, :)' + b1);
        y = sigmoid(W2*a1 + b2);
        
        % Backward pass
        delta = (y - Y(i))*y*(1 - y);
        dW2 = -learningRate*delta*a1';
        db2 = -learningRate*delta;
        delta = W2'*delta.*a1.*(1 - a1);
        dW1 = -learningRate*delta*X(i, :);
        db1 = -learningRate*delta;
        
        % Update the weights
        W1 = W1 + dW1;
        W2 = W2 + dW2;
        b1 = b1 + db1;
        b2 = b2 + db2;
        
        % Calculate the epoch's NMSE
        epochNMSE = epochNMSE + (Y(i) - y).^2;

        %Calculate TSS
        TSS = sum((Y - mean(Y)).^2);

        % Compute the Jacobian and Hessian matrices for the Gauss Newton method
        J = [a1.*X(i, :); zeros(hiddenNodes*outputNodes - inputNodes, 1)];
        Hessian = J'*J;
        
        % Calculate Hessian Inverse
        HessianInverse = inv(Hessian + (0.01 * eye(size(J, 2))));
        HessianInverse = (HessianInverse/278)/100;
        
        % Update weights and biases
        deltaW = [dW1(:); dW2(:); db1; db2];
        W1 = W1 + reshape(deltaW(1:hiddenNodes*inputNodes), hiddenNodes, inputNodes);
        W2 = W2 + reshape(deltaW(hiddenNodes*inputNodes+1:hiddenNodes*inputNodes+outputNodes*hiddenNodes), outputNodes, hiddenNodes);
        b1 = b1 + deltaW(hiddenNodes*inputNodes+outputNodes*hiddenNodes+1:hiddenNodes*inputNodes+outputNodes*hiddenNodes+hiddenNodes);
        b2 = b2 + deltaW(hiddenNodes*inputNodes+outputNodes*hiddenNodes+hiddenNodes+1:end);
    end
    
% Compute the NMSE and error for the epoch
NMSE(epoch) = epochNMSE;
error(epoch) = epochError/size(X, 1);

%if mod(epoch, 20) == 0
    fprintf('Epoch %d: NMSE = %.4f\n', epoch, NMSE(epoch));
%end

end

% Plot the NMSEvalues for each epoch
figure;

% Create subplot for NMSE
plot(1:maxEpochs, NMSE, '-b', 'LineWidth', 1);
xlabel('Epoch');
ylabel('NMSE');
title('NMSE vs Epoch');
grid on;




