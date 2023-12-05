clear
clc

% Load logmIKI trial average data
data = readtable('logmIKI_trialavg.csv');

% Convert group to categorical variable
data.group = categorical(data.group);

% Show data structure
% disp(data);

% Run full ANOVA model
model = fitlme(data, 'logmIKI ~ group');

% Get ANOVA table
anovaTable = anova(model);

% Show ANOVA table
disp(anovaTable);

% Compute Bayes factor for full model
bf_full = bf.anova(data, 'logmIKI ~ group');

% Show Bayes factor
disp(bf_full);

% Split data by group
data0 = data(data.group == '0', :).logmIKI;
data1 = data(data.group == '1', :).logmIKI;
data2 = data(data.group == '2', :).logmIKI;

% Conduct paired ttest for all three pairs: 0-1, 0-2, 1-2
[bf01,p01] = bf.ttest2(data0, data1);
[bf02,p02] = bf.ttest2(data0, data2);
[bf12,p12] = bf.ttest2(data1, data2);

% Show Bayes factors and p values for post-hoc tests
disp(['Bayes factor for difference between group 0 and 1: ', num2str(bf01, '%.4e'), ', p = ', num2str(p01, '%.4e')]);
disp(['Bayes factor for difference between group 0 and 2: ', num2str(bf02, '%.4e'), ', p = ', num2str(p02, '%.4e')]);
disp(['Bayes factor for difference between group 1 and 2: ', num2str(bf12, '%.4e'), ', p = ', num2str(p12, '%.4e')]);



