% reinforcement learning course
% dynamic programming
% clean robot with value iteration

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

% initialize value 
V0 = zeros(7,1); 

% value iteration
Vprev = -inf*ones(7,1); 
V = V0; 
iter = 1; 
while max(abs(V-Vprev)) >= 1e-3
    fprintf('%d-th iteration, current V = %s\n', iter, mat2str(V',5)); 
    Q = R' + gamma * sum(bsxfun(@times,P,permute(V,[2 3 1])),3); 
    fprintf('                 current policy: left = %s, right = %s\n', ...
        mat2str(int8(Q(1,:)>Q(2,:))'), mat2str(int8(Q(1,:)<=Q(2,:))')); 
    Vprev = V; 
    V = max(Q,[],1)'; 
    iter = iter+1; 
end

% % output optimal policy
% for i = 1:length(V)
%     fprintf('S%d optimal action: ',i); 
%     if Vleft(i)>=Vright(i)
%         fprintf('left\n');
%     else
%         fprintf('right\n'); 
%     end
% end