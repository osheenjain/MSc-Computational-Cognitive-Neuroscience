% section A 
tau = 0.015;
dt = tau/50;
Rm = 1.0e+07;
V_thresh = -0.065; 
V_reset = -0.08;
E = -0.07;

%section B 
Ie = 3.5e-09;
T= 0:dt:0.3;
V_hat = zeros(size(T));
S= zeros(size(T));
V_0 = -0.07;
V_hat(1) = V_0;


%section C 
for t=2:length(T)
    if V_hat(t-1)<V_thresh
        V_hat(t)= V_hat(t-1) + (dt/tau) * (E - V_hat(t-1) + Rm * Ie);
    else
        V_hat(t)= V_reset;
        S(t)=1;
    end    
end

% Plotting the membrane potential
plot(T,V_hat)
xlabel('Time (s)')
ylabel('Membrane Potential (V)')

% Compute the firing rate
spike_times = find(S==1);
firing_rate = length(spike_times)/(T(end)-T(1));

firing_rate

