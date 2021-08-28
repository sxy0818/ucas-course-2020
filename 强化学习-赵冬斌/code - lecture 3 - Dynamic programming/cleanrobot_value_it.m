% reinforcement learning course
% dynamic programming
% clean robot evaluate a policy value with iterative method 

% state transition matrix [P(k,i,j)]
%         k: action 
%         i: state 
%         j: successor state 
P = zeros(2,7,7); 
P(1,:,:) = [0.9 0.1 0 0 0 0 0
    0.8 0.1 0.1 0 0 0 0
    0 0.8 0.1 0.1 0 0 0
    0 0 0.8 0.1 0.1 0 0 
    0 0 0 0.8 0.1 0.1 0 
    0 0 0 0 0.8 0.1 0.1
    0 0 0 0 0 0.8 0.2]; 
P(2,:,:) = [0.2 0.8 0 0 0 0 0
    0.1 0.1 0.8 0 0 0 0
    0 0.1 0.1 0.8 0 0 0 
    0 0 0.1 0.1 0.8 0 0 
    0 0 0 0.1 0.1 0.8 0 
    0 0 0 0 0.1 0.1 0.8
    0 0 0 0 0 0.1 0.9]; 

% reward vector
R = [1 0 0 0 0 0 10]'; 

% discount factor
gamma = 0.7; 

% target policy and state transition
pi = zeros(2,7); 
pi(1,:) = 0.5*ones(1,7); 
pi(2,:) = 0.5*ones(1,7);
Ppi = squeeze(sum(bsxfun(@times,pi,P),1)); 

% initialize value
V0 = zeros(7,1); 

% iterative policy evaluation
Vprev = -inf(size(V0)); 
V = V0; 
iter = 1; 
while max(abs(V-Vprev))>1e-3
    fprintf('%d-th iteration, current V = %s\n', iter, mat2str(V',6)); 
    Vprev = V; 
    V = R+gamma*Ppi*V;
    iter = iter+1; 
end

% output value 
fprintf('%d-th iteration, final V = %s\n', iter, mat2str(V',6)); 
