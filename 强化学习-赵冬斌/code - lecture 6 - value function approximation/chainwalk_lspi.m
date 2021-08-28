% chainwalk_lspi.m
% value function approximation
% chain walk problem solved by lspi with Gaussian RBF
clear; close all; clc; 

% state set 
state_set = num2cell(1:50); 

% action set 
act_set = {-1,1}; 

% discount factor 
gamma = 0.9; 

% experience generated from random walk 
sample_s = zeros(1,10000); 
sample_a = zeros(1,10000); 
sample_r = zeros(1,10000); 
sample_sn = zeros(1,10000); 
for i = 1:10000
    if i==1
        s = 1; 
    else
        s = sample_sn(i-1); 
    end
    a = act_set{randi(2,1)}; 
    [sn,r] = chainwalk_model(s,a); 
    sample_s(i) = s; 
    sample_a(i) = a; 
    sample_r(i) = r; 
    sample_sn(i) = sn; 
end

% linear feature combination parameters 
w0 = zeros(10*2,1); 
w = w0; 

% record params history 
whist = [w]; 

% lspi
for i = 1:10
    wold = w; 
    sample_an = policy(sample_sn,w,act_set); 
    X = feature_vector(sample_s,sample_a); 
    Xn = feature_vector(sample_sn,sample_an); 
    w = (X*(X-gamma*Xn)')^(-1) * X * sample_r'; 
    whist = [whist, w]; 
end

% plot params curve 
figure; plot(1:size(whist,2),whist); grid on; 
xlabel('iteration'); ylabel('w'); 

% plot Q value 
s = repmat(cell2mat(state_set),1,length(act_set)); 
a = reshape(repmat(cell2mat(act_set),length(state_set),1),1,[]); 
x = feature_vector(s,a); 
q = reshape(x'*w,[],2)'; 
figure; plot(1:length(state_set),q); grid on; 
xlabel('s'); ylabel('Q'); 
xlim([1,50]); 

% plot policy 
S = repmat(1:0.1:50,1,length(act_set)); 
A = reshape(repmat(cell2mat(act_set),size(S,2)/length(act_set),1),1,[]); 
X = feature_vector(S,A); 
Q = reshape(X'*w,[],2)'; 
[C,opta_ind] = max(Q,[],1); 
opta = cell2mat(act_set(opta_ind)); 
figure; ph = pcolor(1:0.1:50,[-4,4],repmat(opta,2,1)); 
xlabel('s'); 
h = gca; h.YAxis.Visible = 'off';
set(ph, 'LineStyle', 'none'); hold on; box off; axis equal; 
ylim([-4,4]); 



function [sn,r] = chainwalk_model(s,a)
% chain walk model with parallel processing
% input: state s [n*N]
%        action a [m*N]
% output: successor state sn [n*N]
%         reward r [1*n]

% randomize walk direction, same action or reverse action 
dir_front = rand(1,size(a,2))<0.9; 
dir_back = -1*(~dir_front); 
dir = dir_front+dir_back; 

% update successor state 
sn = s+dir*a; 
sn = max(1,min(sn,50)); 

% reward 
r = double(s==10 || s==41); 

end

function a = policy(s,w,act_set)
% policy function, find the argmax Q(s,a)
% input: state s [n*N]
%        approximator params w [(Km)*1]
% output: action a [m*N]

% stack state for all action 
s_stack = repmat(s,1,2); 
a_stack = reshape(repmat(cell2mat(act_set),size(s,2),1),1,[]); 

% feature 
x = feature_vector(s_stack,a_stack); 
q = reshape(x'*w,[],2)'; 

% action 
[Y,a_ind] = max(q,[],1); 
a = cell2mat(act_set(a_ind)); 

end

function x = feature_vector(s,a)
% feature function according to given s and a
% input: state s [n*N]
%        action a [m*N]
% output: feature vector x [(Km)*N]

% RBF center 
c = (1:(50-1)/(10-1):50)'; 

% bandwith
sigma = 4; 

% RBF value
dis = bsxfun(@minus,s,c); 
phi = exp(-dis.^2/2/sigma^2); 
% phi = bsxfun(@rdivide,phi,sum(phi,1)); 

% action index to indicate action position 
a_ind = zeros(2,size(a,2)); 
a_ind(1,:) = (a==-1); 
a_ind(2,:) = (a==1); 
a_stack_ind = repmat(a_ind(:)',10,1); 
a_stack_ind = reshape(a_stack_ind,2*10,[]); 

% stack into feature vector 
x = bsxfun(@times,repmat(phi,2,1),a_stack_ind); 

end




