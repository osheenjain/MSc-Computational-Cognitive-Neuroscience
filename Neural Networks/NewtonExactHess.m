clear
clc

% Original code by Dave Touretzky (1st modified by Nikolay Nikolaev) (2nd Modification by
% Andrew Rice) (Current file further modified by Osheen Jain)
% https://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/matlab/

% Due to an unknown issue, the value of NMSE keeps varying the graph
% changes drastically. The same was happening for the original file. 

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

% Initial Paramaters
Input_patterns = x'; 
Target = y; 
NumNodes = 5; 
Output_patterns = Target;

[NINPUTS,NPATS] = size(Input_patterns); 
[NOUTPUTS,NP] = size(Target);

LearnRate = 1.0; 
Momentum = 0;  
deltaW1 = 0; 
deltaW2 = 0;

Inputs1 = [ones(1,NPATS); Input_patterns];

Weights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NumNodes)-0.5);

Vweights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Vweights2 = 0.5*(rand(1,1+NumNodes)-0.5);

%Vweights1 = randi([0, 1], NumNodes,1+NINPUTS);
%Vweights2 = randi([0, 1], 1,1+NumNodes);

TSS_Limit = 0.2;

Weights = [reshape(Weights1, 1, []), reshape(Weights2, 1, [])];

Out = zeros(1,NPATS); 
Epoch = 250
variance = var(Target); %For calculating NMSE

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
    NMSE(Epoch) = MSE(Epoch)/variance;

    %% Weight Updates

    Weights1 = deltaW1 + Weights1;
    Weights2 = deltaW2 + Weights2;
                               
    fprintf('Epoch %3d:  NMSE = %f\n',Epoch,NMSE);
    if TSS < TSS_Limit, break, end

 end

 plot(year(idim+1:288),Target,year(idim+1:288),Output_patterns)
title('Sunspot Data')

plot(NMSE, '--', 'LineWidth', 1);
