clear all
close all
clc

% Load sunspot data
load sunspot.dat
year = sunspot(:, 1);
sunspotNums = sunspot(:, 2);

% normalizes the sunspot data 
sunspotNums = (sunspotNums - min(sunspotNums)) / (max(sunspotNums) - min(sunspotNums)) * 2 - 1;

% create a matrix of lagged values for a time series vector
TimeSeriesVector = sunspotNums';
idim = 10; % input dimension
odim = length(TimeSeriesVector) - idim; % output dimension

x = zeros(odim, idim);
y = TimeSeriesVector(idim+1:end);

for i = 1:odim
    x(i, :) = TimeSeriesVector(i:i+idim-1)';
end

% Set initial parameters
Input_patterns = x'; % input patterns
Target = y; % desired output patterns
Output_patterns = Target; % output patterns

NumNodes = 5; % number of hidden nodes
variance = var(Target);

% Get the number of input and output patterns
[NINPUTS, NPATS] = size(Input_patterns);
[NOUTPUTS, NP] = size(Target);

% Set learning rate, momentum, and other parameters
LearnRate = 0.2;
Momentum = 0;
TSS_Limit = 0.02;
LR = 0.001;
lambda=0.001; % Hessian regularization Parameter

% Add bias to input patterns and initialize weights randomly
Inputs1 = [ones(1, NPATS); Input_patterns]';
Weights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NumNodes)-0.5); 

% Flatten the weight matrices into a single weight vector
Weights = [reshape(Weights1, 1, []), reshape(Weights2, 1, [])];

% Initialize the nodes and output patterns
Out = 1;

Epoch = 250;

% Calling the function for Approx Hessin
[OutStore, NMSE, MSE] = NewtonApproxHess(Inputs1, Output_patterns, NumNodes, lambda, variance);

% Calling the function for Backpropogation
[NMSE1, MSE1, out_dimension] = backpropagation(x, y, LR, Momentum);

% Calling the function for Exact Hessian
[NMSE2] = NewtonExactHess(x, y, Weights1, Weights2, Momentum, variance, Weights);

% Create a figure
figure;

% Plot RMSE using a solid blue line
semilogy(NMSE, '-b', 'LineWidth', 2);

% Add a legend entry for RMSE
hold on;
leg = legend('NMSE (backpropagation)');
set(leg, 'FontSize', 14);

% Plot RMSE1 using a dashed red line
semilogy(NMSE1, '--r', 'LineWidth', 2);

% Add a legend entry for RMSE1
leg = legend('NMSE (backpropagation)', 'RMSE1 (ApproxHessian)');
set(leg, 'FontSize', 14);

% Plot RMSE2 using a dotted green line
semilogy(NMSE2, ':g', 'LineWidth', 2);

% Add a legend entry for RMSE2
leg = legend('NMSE (Approx Hessian)', 'NMSE1 (Backpropagation)', 'NMSE2 (Exact Hessian)');
set(leg, 'FontSize', 14);

% Add axis labels and a title
xlabel('Epoch', 'FontSize', 16);
ylabel('RMSE', 'FontSize', 16);
title('Performance Comparison', 'FontSize', 18);

function [OutStore, NMSE, MSE] = NewtonApproxHess(Inputs1, Output_patterns, NumNodes, lambda, variance)

Out = 1;
NPATS = size(Inputs1,1);
Weights1=rand(NumNodes,size(Inputs1,2));
Weights2=rand(1,NumNodes+1);

for Epoch=1:250
    % Clearing of delta Weights for each Epoch
    deltaW1=zeros(5,11);
    deltaW2=zeros(1,6);

    % Getting the Jacobian with Beta=1,HiddenBeta=1
    if Epoch==1
        for p=1:NPATS   
            %% Forward Propogation
            for i=1:NumNodes
                activation=0;
                for j=1:size(Inputs1,2)
                    activation = activation + Weights1(i,j)*Inputs1(p,j);
                    hiddenLay(i) = 1.0/(1.0 + exp(-activation));
                end                    
            end
            if p==1
                hiddenLay = [ones(1) hiddenLay];
            end

            % Hidden to Output
            for i = 1: Out
                SigOut = 0;
                for w = 1:NumNodes+1
                    SigOut =  hiddenLay(w)*Weights2(1,w) + SigOut;
                end
            end
            SumOut = SigOut;

            %% Backward Propogation
            %Error in Output Nodes
            for i = Out:-1:1
                err = (Output_patterns(p) - SumOut);
                BetaOut =  1;
            end

            % Error in Hidden Nodes
            for  j = NumNodes+1:-1:1
                    bHidden(j) = 0;
                    for l = 1: Out 
                        bHidden(j) = Weights2(j) * BetaOut(l);
                        bHidden(j) = hiddenLay(j) * (1-hiddenLay(j))* bHidden(j);
                    end
            end
            %bHidden=1;
            %% Jacobian Time
            % Input to Hidden
            for i =1:size(Inputs1,2)
                for j=1:NumNodes
                JbI_O(j,i)=bHidden(1,j)*Inputs1(p,i);
                end
            end
            % Hidden to Output
            for jj=1:NumNodes+1
                jbH_O(1,jj)= (BetaOut)*hiddenLay(jj);
            end
            Jtest = reshape(JbI_O',[],1);
            Jacob(p,:)=[Jtest' jbH_O(1,:)]; 
        end
    end                  
                    
  %% Second NN
    for p=1:NPATS    
        %% Forward Propogation
        for i=1:NumNodes
            activation=0;
            for j=1:size(Inputs1,2)
                activation = activation + Weights1(i,j)*Inputs1(p,j);
                hiddenLay(i) = 1.0/(1.0 + exp(-activation)); 
            end
        end
        
        % Hidden to Output
        for i = 1: Out
            SigOut = 0;
            for w = 1:NumNodes+1
                SigOut =  hiddenLay(w)*Weights2(1,w) + SigOut;
            end
        end
        SumOut = SigOut;
        OutStore(p)=SumOut;
        
        %% Backward Propogation
        % Error in Output Nodes
        for i = Out:-1:1
            err = (Output_patterns(p) - SumOut);
            BetaOut =  err;
            Error(p)=BetaOut;
        end

        % Error in Hidden Nodes
        for  j = NumNodes+1:-1:1
            bHidden(j) = 0;
            for l = 1: Out 
                bHidden(j) = Weights2(j) * BetaOut(l);
                bHidden(j) = (hiddenLay(j) * (1-hiddenLay(j))* bHidden(j));
            end
        end
        
        if Epoch>1
        %% Jacobian Time
        
            % Input to Hidden
            for i =1:size(Inputs1,2)
                for j=1:NumNodes
                JbI_O(j,i)=(bHidden(1,j)*Inputs1(p,i));
                end
            end
            % Hidden to Output
            for jj=1:NumNodes+1
                jbH_O(1,jj)= (BetaOut)*hiddenLay(jj);
            end
            Jtest = reshape(JbI_O',[],1);
            Jacob(p,:)=[Jtest' jbH_O(1,:)]; 
        end
               
     % Update for Input to Hidden
        for i = 1:NumNodes            
            for j = 1:size(Inputs1,2)
               deltaW1(i,j)=deltaW1(i,j)+bHidden(i)*Inputs1(p,j);
            end            
        end
        
        for i = 1:Out
            for j = 1:NumNodes   
               deltaW2(i,j)=deltaW2(i,j)+BetaOut(i)*hiddenLay(j);
            end  
        end      
            
    end
    %% Jacobian Calculation
    % Update for Hidden to Output
     
    Hess=(Jacob'*Jacob+lambda*eye(61));
    invDiagHess=diag(1./(Hess))/NPATS/5;
    LR1=invDiagHess(1:55);
    LR11 = reshape(LR1,[],11);
    LR2=invDiagHess(56:61);
     
    TSS =   sum((Error).^2);
    MSE(Epoch) =  sum((Error).^2);
    NMSE(Epoch) = MSE(Epoch)/variance;


    %% Weight update Batch Mode
    for i = 1:NumNodes            
        for j = 1:size(Inputs1,2)
           Weights1(i,j) = Weights1(i,j)+ LR11(i,j) * deltaW1(i,j); 
        end            
    end 
    for i = 1:Out
        for j = 1:NumNodes+1  
           Weights2(i,j) = Weights2(i,j) + LR2(i) * deltaW2(i,j); 
        end  
    end    
    
    Error;
    lambda;

% if mod(Epoch, 10) == 0 || TSS < TSS_Limit
%     fprintf('Epoch %3d:  Error = %f\n', Epoch, TSS);
% end

end
    
end


function [NMSE1, MSE1, out_dimension] = backpropagation(x, y, LR, Momentum)

%BACKPROPAGATION Trains a neural network using backpropagation
%   Inputs:
%       Inputs1: input data (matrix)
%       Target: target data (matrix)
%       Weights1: weights for the first layer (matrix)
%       Weights2: weights for the second layer (matrix)
%       Epoch: number of Epoch (scalar)
%       LR: learning rate (scalar)
%       Momentum: momentum (scalar)
%
%   Outputs:
%       MSE: mean squared error (vector)
%       out_dimension: predicted output (matrix)


% Set initial parameters
Input_patterns = x'; % input patterns
Target = y; % desired output patterns
Output_patterns = Target; % output patterns

NumNodes = 5; % number of hidden nodes
variance = var(Target);

% Get the number of input and output patterns
[NINPUTS, NPATS] = size(Input_patterns);
[NOUTPUTS, NP] = size(Target);

% Set learning rate, momentum, and other parameters
TSS_Limit = 0.02;
lambda=0.001; % Hessian regularization Parameter
Epoch = 250;

% Add bias to input patterns and initialize weights randomly
Inputs1 = [ones(1, NPATS); Input_patterns];
Weights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NumNodes)-0.5); 

% Flatten the weight matrices into a single weight vector
Weights = [reshape(Weights1, 1, []), reshape(Weights2, 1, [])];

% Initialize the nodes and output patterns
Out = 1;

% Initialize variables
NPATS = size(Inputs1, 2);
NP = size(Target, 1) * size(Target, 2);
deltaW1 = 0;
deltaW2 = 0;
MSE1 = zeros(1, Epoch);

for Epoch = 1:Epoch
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
    MSE1(Epoch) = TSS/NP;
    NMSE1(Epoch) = MSE1(Epoch)/variance;

    % Plot predicted and desired values subplot

end

end

function [NMSE2] = NewtonExactHess(x, y, Weights1, Weights2, Momentum, variance, Weights)

Input_patterns = x'; 
Target = y; 
NumNodes = 5; 
Output_patterns = Target;

[NINPUTS,NPATS] = size(Input_patterns); 
[NOUTPUTS,NP] = size(Target);

Inputs1 = [ones(1,NPATS); Input_patterns];

Vweights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Vweights2 = 0.5*(rand(1,1+NumNodes)-0.5);

Epoch = 250;

for Epoch = 1:250 

    % Forward propagation
    NetIn1 = Weights1 * Inputs1; % aj
    RNetIn1 = Vweights1 * Inputs1; % R{aj}
    Hidden=1-2./(exp(2*NetIn1)+1); % zj
    RHidden = Hidden .* (1 - Hidden) .* (RNetIn1); % R{zj} 
    Inputs2 = [ones(1,NPATS); Hidden];
    RInputs2 = [ones(1,NPATS); RHidden]; 
    NetIn2 = Weights2 * Inputs2; % yk
    RNetIn2 = (Weights2 * RInputs2) + (Vweights2* Inputs2); % R{yk} 
    Out = NetIn2;  prnout=Out; ROut = RNetIn2; Rprnout = ROut; % R{7k}
    Error = Target - Out; RError = Target - ROut; % &k
    TSS = sum(sum( Error.^2 )); RTSS = sum(sum( RError.^2 ));
    TSSplot(Epoch,1) = TSS; epochplot(Epoch,1) = Epoch;
    
    %% Backpropogation

    Beta(1,1:size(Out,2)) = 1;
    RBeta = Error; 
    bperr = ( Weights2' * Beta );
    HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);
    RPrime = (1 - Hidden.*Hidden);% g'(aj)
    RPrime2 = -2.0 * Hidden .* RPrime; % g"(aj)
    
    Rbperr1 = Vweights2'* RBeta; %sum(vkj,&k)
    Rbperr2 = Weights2' * ROut; % sum(wkj, R{&k})
    Rbperr3 = ( Weights2' * RBeta );

    RHIDBeta1 = RPrime.* Rbperr3(1:end-1,:);  % &j
    RHIDBeta2 = RPrime2.* RNetIn1.*Rbperr3(1:end-1,:);
    RHIDBeta3 = RPrime.*Rbperr1(1:end-1,:);% g' * sum(vkj,&k)
    RHIDBeta4 =  RPrime.*(Weights2(:, 1:end-1)* Hidden.*(1 -Hidden).* ROut);%g' * sum(wkj, R{&k})
    RHiddenBeta = RHIDBeta2 + RHIDBeta3 + RHIDBeta3; %R{&j}

    % ((1-(1/1+e^-x))* ((1/1+e^-x) * 1-(1/1+e^-x))) + (((1/1+e^-x)*
    % 1-(1/1+e^x))) = sigmoid(x) - 3(sigmoid(x))^2 + 2(sigmoid(x))^3;
    
    %% Exact Hessian

    RY = ROut;
    RZ = RInputs2;                           
    HW1 = Inputs1 * RHiddenBeta';
    HW2i = (Inputs2 * RY');
    HW2ii = ( RZ * RHIDBeta1');
    HW2iii = HW2i + HW2ii;
    HW2= sum(HW2iii, 2)';
    HW2 = HW2 + (RBeta* Inputs2');

    VtH = [HW2(:)', HW1(:)'];
    Hessian = zeros(size(Weights,2));

%     for V = eye(size(Weights,2))
%         Hessian(find(V),:)  = VtH;
%     end
   
    Hessian = VtH' * VtH;
    
    Hessian = inv(Hessian + (0.01 * eye(61)));
    Hessian = (Hessian/278)/100;

    TSS =   sum((Error).^2);
    MSE1(Epoch) =  sum((Error).^2);
    NMSE1(Epoch) = MSE1(Epoch)/variance;
                                    
    %% Second pass with Hessian

    % Forward propagation
    
    NetIn1 = Weights1 * Inputs1;
    Hidden=1-2./(exp(2*NetIn1)+1); %Hidden = tanh( NetIn1 );
    Inputs2 = [ones(1,NPATS); Hidden];
    NetIn2 = Weights2 * Inputs2;
    Out = NetIn2;  Output_patterns=Out;
    
    % Backward propagation of errors
    Error = Target - Out;
    TSS = sum(sum( Error.^2 )); % sum(sum(E.*E));
    Beta = Error;
    bperr = ( Weights2' * Beta );
    HiddenBeta = (1.0 - Hidden .^2 ) .* bperr(1:end-1,:);

    dW2 = Beta * Inputs2';
    dW1 = HiddenBeta * Inputs1';

    %% Delta Updates with Hessian

    for dw2Col = 1:size(dW2,2)%6
        for dw2Row = 1:size(dW2,1)%1
            deltaW2(dw2Row,dw2Col) = Momentum + dW2(dw2Row,dw2Col) * ...
                Hessian(dw2Col,dw2Col);
        end
    end
    Count = 0;
    for dw1Row = 1:size(dW1,1)%11
        for dw1Col = 1:size(dW1,2)%55  
            Count = Count + 1;
            deltaW1(dw1Row,dw1Col) = Momentum + dW1(dw1Row,dw1Col) * ...
                Hessian(11-dw1Row+ dw1Col- 5, 11-dw1Row+ dw1Col- 5);
        end
    end

    TSS =   sum((Error).^2);
    MSE(Epoch) =  sum((Error).^2);
    NMSE2(Epoch) = MSE(Epoch)/variance;

    %% Weight Updates

    Weights1 = deltaW1 + Weights1;
    Weights2 = deltaW2 + Weights2;
                               
    fprintf('Epoch %3d:  NMSE = %f\n',Epoch,NMSE2);
    
end
end
