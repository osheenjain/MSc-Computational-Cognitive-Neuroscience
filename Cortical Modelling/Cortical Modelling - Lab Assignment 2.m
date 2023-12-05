% Define parameters
tau = 0.02; % time constant
A = 0.5; % Alpha-coupling parameter for neuron 1
B = 0.5; % Beta-coupling parameter for neuron 2
U = 0.5; % mean of input current
SD = 0.1; % standard deviation of input current
h = 0.00001; % time step
tsim = 0.25; % simulation time

% Initialize vectors
t = 0:h:tsim;
n = length(t);
r1 = zeros(n, 1);
r2 = zeros(n, 1);
I1 = normrnd(U, SD, n, 1);
I2 = normrnd(U, SD, n, 1);

% Euler's method
for i = 2:n
    r1(i) = r1(i-1) + h*(-r1(i-1) + A*r2(i-1) + I1(i-1))/tau;
    r2(i) = r2(i-1) + h*(-r2(i-1) + B*r1(i-1) + I2(i-1))/tau;
end

% Plot results
plot(t,r1,'k',t,r2,'g--', LineWidth=2);
xlabel('Time');
ylabel('Rate');
title(sprintf('Firing Rate: r1 = %.1f (t=%.2f s), r2 = %.1f (t=%.2f s)', r1(end), t(end), r2(end), t(end)), 'FontSize', 12);
legend('Neuron 1','Neuron 2','Location', 'best','FontSize', 14);