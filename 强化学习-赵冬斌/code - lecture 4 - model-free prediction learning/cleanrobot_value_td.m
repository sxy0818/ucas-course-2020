% reinforcement learning course
% model-free prediction learning
% clean robot evaluate a policy value with temporal-difference method

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

% learning length, initial state, learning rate
N = 10000; 
s = 1; 
alpha = 0.01; 

% initialize value
V = zeros(7,1); 

% record value 
Vhist = zeros(7,N); 

% td learning 
for t = 1:N
    sn = randsrc(1,1,[1:7; Ppi(s,:)]); 
    delta = R(s)+gamma*V(sn)-V(s); 
    V(s) = V(s)+alpha*delta; 
    s = sn; 
    Vhist(:,t) = V;
end

% Êä³ö½á¹û 
figure; plot(1:N,Vhist,'linewidth',2); xlabel('step'); ylabel('V(s)'); 
Vtrue = (eye(7)-gamma*Ppi)^(-1)*R; 
figure; plot(1:N,sqrt(sum((bsxfun(@minus,Vhist,Vtrue)).^2,1))); xlabel('step'); ylabel('|V-V^{\pi}|'); 

% figure; plot(1:N,sqrt(sum((bsxfun(@minus,Vhist0005,V)).^2,1)),'linewidth',2,'color','b'); xlabel('step'); ylabel('|V-V^{\pi}|'); hold on; 
% plot(1:N,sqrt(sum((bsxfun(@minus,Vhist001,V)).^2,1)),'linewidth',2,'color','r');
% plot(1:N,sqrt(sum((bsxfun(@minus,Vhist005,V)).^2,1)),'linewidth',2,'color','g');
% legend('\alpha=0.005','\alpha=0.01','\alpha=0.05');

