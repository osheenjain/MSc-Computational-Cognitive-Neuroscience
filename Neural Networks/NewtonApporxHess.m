

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
Target1 = y; % desired output patterns
Output_patterns = Target1; % output patterns

NumNodes = 5; % number of hidden nodes
variance = var(Target1);

% Get the number of input and output patterns
[NINPUTS, NPATS] = size(Input_patterns);
[NOUTPUTS, NP] = size(Target1);

% Set learning rate, momentum, and other parameters
Momentum = 0;
TSS_Limit = 0.02;
lambda=0.001; % Hessian regularization Parameter

% Add bias to input patterns and initialize weights randomly
Inputs1 = [ones(1, NPATS); Input_patterns]';
Weights1 = 0.5*(rand(NumNodes,1+NINPUTS)-0.5);
Weights2 = 0.5*(rand(1,1+NumNodes)-0.5); 

% Flatten the weight matrices into a single weight vector
Weights = [reshape(Weights1, 1, []), reshape(Weights2, 1, [])];

% Initialize the nodes and output patterns
Out = 1;
                      
for Epoch=1:250
    % Clearing of delta Weights for each epoch
    deltaW1=zeros(5,11);
    deltaW2=zeros(1,6);

    
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
                    
  %% Second Pass
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
        
            %% Jacobian 
        
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
    %% Calculating Approx Hessian
    % Update for Hidden to Output
     
    Hess=(Jacob'*Jacob+lambda*eye(61));
    invDiagHess=diag(1./(Hess))/NPATS/5;
    LR1=invDiagHess(1:55);
    LR11 = reshape(LR1,[],11);
    LR2=invDiagHess(56:61);
     
    %% Calculating TSS, MSE, and NMSE
    TSS =   sum((Error).^2);
    MSE1(Epoch) =  sum((Error).^2);
    NMSE1(Epoch) = MSE1(Epoch)/variance;


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
    
% Plot to see if it fits the Sunspot data
subplot(2, 1, 1);
plot(year(11:end), Target1, 'DisplayName', 'Target', LineWidth=1);
hold on;
plot(year(11:end), OutStore, 'DisplayName', 'Predicted', LineWidth=1);
xlabel('Year');
ylabel('sunspots');
legend('Location', 'best', 'FontSize', 12);
title('Sunspot prediction with Gauss Newton');
grid on;

% Plot to see the MSE
subplot(2, 1, 2);
% Plot MSE and NMSE subplot
subplot(2,1,2);
plot(1:Epoch, MSE1, 'DisplayName', 'MSE', 'LineWidth', 1);
hold on;
plot(1:Epoch, NMSE1, 'DisplayName', 'NMSE', 'LineWidth', 1);
xlabel('Epoch');
ylabel('Error');
legend('Location', 'best', 'FontSize', 12);
title('MSE and NMSE with with Gauss Newton');
