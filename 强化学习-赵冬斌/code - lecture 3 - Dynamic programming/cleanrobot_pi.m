% reinforcement learning course
% dynamic programming
% clean robot with policy iteration and matrix for policy evaluation

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

% initial policy [pi(k,i)]
%        k: action 
%        i: state
pi0 = zeros(2,7); 
pi0(1,:) = 0.5*ones(1,7); 
pi0(2,:) = 0.5*ones(1,7); 

% policy iteration
piprev = inf(size(pi0)); 
pi = pi0; 
iter = 1; 
while max(abs(pi(:)-piprev(:))) > 1e-3
    fprintf('%d-th iteration, current policy: left = %s, right = %s \n', ...
        iter, mat2str(pi(1,:),5), mat2str(pi(2,:),5)); 
    Ppi = bsxfun(@times,pi(1,:)',squeeze(P(1,:,:)))+...
        bsxfun(@times,pi(2,:)',squeeze(P(2,:,:))); 
    V = (eye(7)-gamma*Ppi)^(-1)*R; 
    fprintf('                 current value = %s\n', mat2str(V',5)); 
    Q = R' + gamma * sum(bsxfun(@times,P,permute(V,[2 3 1])),3); 
    piprev = pi;
    pi = zeros(2,7); 
    [C,I] = max(Q,[],1); 
    pi(1,I==1) = 1; 
    pi(2,I==2) = 1; 
    iter = iter+1; 
end

% output optimal policy 
fprintf('%d-th iteration, optimal policy: left = %s, right = %s \n', ...
        iter, mat2str(pi(1,:),5), mat2str(pi(2,:),5)); 
