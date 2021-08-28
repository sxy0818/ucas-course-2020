% motor_lspi.m
% value function approximation
% LSPI for continuous state space with finite action set
clear; close all; clc; 

% state space 
max_s1 = pi; 
min_s1 = -pi; 
max_s2 = 16*pi; 
min_s2 = -16*pi; 

% action set 
act_set = {-10,0,10}; 

% discount factor 
gamma = 0.95; 

% experience generated from random 
sample_s = [min_s1;min_s2]+...
    bsxfun(@times,[max_s1-min_s1;max_s2-min_s2],rand(2,7500)); 
sample_a = cell2mat(act_set(randi(3,1,7500))); 
[sample_sn,sample_r] = motor_model(sample_s,sample_a); 

% linear feature combination parameters 
w0 = zeros(9*9*3,1); 
w = w0; 

% record params history 
whist = [w]; 

% lspi
for i = 1:20
    wold = w; 
    sample_an = policy(sample_sn,w,act_set); 
    X = feature_vector(sample_s,sample_a,act_set); 
    Xn = feature_vector(sample_sn,sample_an,act_set); 
    w = (X*(X-gamma*Xn)')^(-1) * X * sample_r'; 
    whist = [whist, w]; 
end

% plot params curve 
figure; plot(1:size(whist,2),whist); grid on; 
xlabel('iteration'); ylabel('w'); 

% plot Q shape under a = 0
s1 = min_s1:(max_s1-min_s1)/100:max_s1; 
s2 = min_s2:(max_s2-min_s2)/100:max_s2; 
[S1,S2] = meshgrid(s1,s2); 
s = [S1(:)';S2(:)']; 
a = reshape(repmat(act_set{2},size(s,2),1),1,[]); 
x = feature_vector(s,a,act_set); 
qa0 = x'*w; 
figure; mesh(S1,S2,reshape(qa0,size(S1))); 
xlabel('\alpha [rad]');
ylabel('\alpha'' [rad/s]');
zlabel('Q (\alpha,\alpha'',0)');
xlim([-pi pi]); ylim([-16*pi 16*pi]);

% plot policy 
S = repmat(s,1,length(act_set)); 
A = reshape(repmat(cell2mat(act_set),size(S,2)/length(act_set),1),1,[]); 
X = feature_vector(S,A,act_set); 
Q = reshape(X'*w,[],length(act_set))'; 
[C,opta_ind] = max(Q,[],1); 
opta = cell2mat(act_set(opta_ind)); 
figure; ph = pcolor(S1,S2,reshape(opta,size(S1))); 
set(ph, 'LineStyle', 'none'); hold on; box on;
xlabel('\alpha [rad]');
ylabel('\alpha'' [rad/s]');
title('\pi(\alpha,\alpha'') [V]');
gs.schemec = 'k';
gs.innerc = [1 1 1]; gs.centerc = 0.25*[1 1 1];
gs.cm = gray(128); gs.cm = gs.cm(33:end,:);
gs.mesh = {'EdgeColor', [.3 .3 .3]};
colormap(gs.cm);






function [sn,r] = motor_model(s,a)
% motor system model, capable of parallel processing data 
% input: state s [n*N]
%        action a [m*N]
% output: successor state sn [n*N]
%         reward r [1*N]

% compute successor state
sn = [1 0.0049;0 0.9540]*s+[0.0021;0.8505]*a;

% compute reward 
r = [-5,-0.01,-0.01]*[s.^2; a.^2]; 

end

function a = policy(s,w,act_set)
% policy function, find the argmax Q(s,a)
% input: state s [n*N]
%        approximator params w [(Km)*1]
% output: action a [m*N]

% stack state for all action 
s_stack = repmat(s,1,length(act_set)); 
a_stack = reshape(repmat(cell2mat(act_set),size(s,2),1),1,[]); 

% feature 
x = feature_vector(s_stack,a_stack,act_set); 
q = reshape(x'*w,[],length(act_set))'; 

% action 
[Y,a_ind] = max(q,[],1); 
a = cell2mat(act_set(a_ind)); 

end

function x = feature_vector(s,a,act_set)
% feature function according to given s and a
% input: state s [n*N]
%        action a [m*N]
% output: feature vector x [(Km)*N]

% RBF center 
c1 = -pi:2*pi/(9-1):pi; 
c2 = -16*pi:16*pi*2/(9-1):16*pi; 
[C1,C2] = meshgrid(c1,c2); 
c = [C1(:)'; C2(:)']; 

% bandwith
sigma = [2*pi/8; 32*pi/8]; 

% RBF value
dis = bsxfun(@minus,permute(s,[1 3 2]),c); 
ndis = permute(sum(bsxfun(@rdivide,dis.^2,sigma.^2),1),[2 3 1]); 
phi = exp(-1/2*ndis); 
phi = bsxfun(@rdivide,phi,sum(phi,1)); 

% action index to indicate action position 
a_ind = zeros(length(act_set),size(a,2)); 
for i = 1:length(act_set)
    a_ind(i,:) = (a==act_set{i}); 
end
a_stack_ind = repmat(a_ind(:)',size(c,2),1); 
a_stack_ind = reshape(a_stack_ind,length(act_set)*size(c,2),[]); 

% stack into feature vector 
x = bsxfun(@times,repmat(phi,length(act_set),1),a_stack_ind); 

end



