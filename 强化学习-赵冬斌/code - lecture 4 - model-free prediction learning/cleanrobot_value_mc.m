% reinforcement learning course
% model-free prediction learning
% clean robot evaluate a policy value with Monte-Carlo method

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

% sample times 
N = 10000; 
G = zeros(7,N); 

% episode length
M = 100; 

% simulate episode
for k = 1:N
    for i = 1:7
        s = i; 
        r = 0; 
        for t = 1:M
            r = r+gamma^(t-1)*R(s); 
            sn = randsrc(1,1,[1:7; Ppi(s,:)]); 
            s = sn; 
        end
        G(i,k) = r; 
    end
%     if mod(k,100)==0
%         fprintf('sample %d times, mean V(1) = %s\n',k,mat2str(mean(G(:,1:k),2)'));
%     end
end

% Êä³ö½á¹û 
fprintf('after %d samples, V = %s\n',1, mat2str(mean(G(:,1:1),2)',5)); 
fprintf('after %d samples, V = %s\n',10, mat2str(mean(G(:,1:10),2)',5)); 
fprintf('after %d samples, V = %s\n',50, mat2str(mean(G(:,1:50),2)',5)); 
fprintf('after %d samples, V = %s\n',100, mat2str(mean(G(:,1:100),2)',5)); 
fprintf('after %d samples, V = %s\n',500, mat2str(mean(G(:,1:500),2)',5)); 
fprintf('after %d samples, V = %s\n',1000, mat2str(mean(G(:,1:1000),2)',5)); 
fprintf('after %d samples, V = %s\n',5000, mat2str(mean(G(:,1:5000),2)',5)); 
fprintf('after %d samples, V = %s\n',10000, mat2str(mean(G(:,1:10000),2)',5)); 

