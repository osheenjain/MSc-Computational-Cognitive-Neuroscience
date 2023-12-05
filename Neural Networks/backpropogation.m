clc
clear

% Load sunspot data
load sunspot.dat

% Normalize and extract time series data
year = sunspot(:, 1);
sunspotNums = sunspot(:, 2);

% normalizes the sunspot data 
sunspotNums = (sunspotNums - min(sunspotNums)) / (max(sunspotNums) - min(sunspotNums)) * 2 - 1;
TimeSeriesVector = sunspotNums';

% Define input and output dimensions
input_dimension = 10;
output_dimension = length(TimeSeriesVector) - input_dimension;

% Initialize input and output arrays
y = TimeSeriesVector(input_dimension+1:end);
x = zeros(output_dimension, input_dimension);
for i = 1:input_dimension
    x(:, i) = TimeSeriesVector(i:output_dimension+i-1);
end

Patterns = x';
NINPUTS = input_dimension; NPATS = output_dimension; NOUTPUTS = 1; NP = output_dimension;
Target = y; NHIDDENS = 5; out_dimension = Target;
LR = 0.001; Momentum = 0;  deltaW1 = 0; deltaW2 = 0;
Inputs1 = [Patterns; ones(1, NPATS)];
Weights1 = 0.5 * (rand(NHIDDENS, 1+NINPUTS)-0.5);
Weights2 = 0.5 * (rand(1, 1+NHIDDENS)-0.5);
TSS_Limit = 0.02;
Epoch = 250;

variance = var(Target);

% Initialize array to store MSE values
mse = zeros(1, Epoch);

for epoch = 1:Epoch
    % Feedforward
    NetIn1 = Weights1 * Inputs1;
    Hidden = (1 - 2 ./ (exp(2*NetIn1)+1));
    Inputs2 = [Hidden; ones(1, NPATS)];
    NetIn2 = Weights2 * Inputs2;
    Out = NetIn2; out_dimension = Out;
    Error = Target - Out;
    TSS = sum(sum((Error.^2)));
    
    % Backpropagation
    Beta = Error;
    bperr = Weights2' .* Beta;
    HiddenBeta = ((1 - Hidden.^2) .* bperr(1:end-1,:));
    dW2 = Beta * Inputs2';
    dW1 = HiddenBeta * Inputs1';
    deltaW2 = LR * dW2 + Momentum * deltaW2;
    deltaW1 = LR * dW1 + Momentum * deltaW1;
    Weights2 = Weights2 + deltaW2;
    Weights1 = Weights1 + deltaW1;
    
    % Calculate MSE and store it in array
    MSE(epoch) = TSS/NP;
    NMSE(epoch) = MSE(epoch)/variance;
    
    fprintf('Epoch: %d Error: %f\n', epoch, TSS);
end

% Create figure with subplots
figure;

% Plot predicted and desired values subplot
subplot(2,1,1);
plot(year(11:end), Target, 'DisplayName', 'Target', LineWidth=1);
hold on;
plot(year(11:end), out_dimension, 'DisplayName', 'Predicted', LineWidth=1);
xlabel('Year');
ylabel('sunspots');
legend('Location', 'best', 'FontSize', 12);
title('Sunspot prediction with Backpropagation');

subplot(2,1,2);
subplot(2,1,2);
plot(1:Epoch, MSE, 'DisplayName', 'MSE', 'LineWidth', 1);
hold on;
plot(1:Epoch, NMSE, 'DisplayName', 'NMSE', 'LineWidth', 1);
xlabel('Epoch');
ylabel('Error');
legend('Location', 'best', 'FontSize', 12);
title('MSE and NMSE with Backpropagation Algorithm');