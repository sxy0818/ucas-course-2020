% motor DC control 
% policy gradient 
% REINFORCE algorithm 

clear; close all; clc; 

% problem setting 
s_max = [pi; 16*pi];        % state space limit
s_min = [-pi; -16*pi]; 
actions = {-10,0,10};       % finite action set
gamma = 0.95;               % discount 
motor_func = @(s,a) max(s_min, min(s_max, [1 0.0049;0 0.9540]*s+[0.0021;0.8505]*a));
reward_func = @(s,a) [-5,-0.01,-0.01]*[s.^2; a.^2]; 

% RBF basis function 
c1 = -pi:2*pi/(9-1):pi; 
c2 = -16*pi:16*pi*2/(9-1):16*pi; 
[C1,C2] = meshgrid(c1,c2); 
c = [C1(:)'; C2(:)']; 
B = repmat((diag([1/(2*pi/8)^2; 1/(32*pi/8)^2])), [1 1 size(c,2)]); 

% policy params 
theta = zeros(size(c,2) * length(actions), 1); 

% check 
ssample = rand(2,1); 
asample = [0 1 0]'; 
check(@(x) policy(ssample,asample,c,B,x), rand(size(theta))); 

% policy gradient params
iter_num = 100;         % policy gradient iteration 
episode_num = 1;      % policy evaluate episode
T = 300;                % policy roll steps 
alpha = 0.05;

% store information 
theta_traj = zeros(length(theta), iter_num); 
J_traj = zeros(1, iter_num); 
figure(12); xlim([0 iter_num]); 

for iter = 1:iter_num
    sdata = []; 
    adata = []; 
    Gdata = []; 

    for episode = 1:episode_num
        s0 = [-pi; 0];          % fixed initial state 
%         s0 = rand(2,1).*(s_max-s_min) + s_min;      % random initial state
        % collect online trajectory
        [straj, atraj, rtraj] = rolling_model(s0, T, actions, ...
            motor_func, ...
            @(s) policy0(s,c,B,theta), ...
            reward_func); 
        Gtraj = zeros(1,T); 
        for i = 1:T
            Gtraj(1:T-(i-1)) = Gtraj(1:T-(i-1)) + gamma^(i-1) * rtraj(i:T); 
        end
        sdata = [sdata straj(:,1:floor(T/2))]; 
        adata = [adata atraj(:,1:floor(T/2))]; 
        Gdata = [Gdata Gtraj(:,1:floor(T/2))]; 
        J_traj(iter) = J_traj(iter) + Gtraj(1); 
    end

    % policy gradient 
    [~, N] = size(sdata); 
    [p,dp] = policy(sdata,adata,c,B,theta);     % 1*N, (km)*N
    lrdata = bsxfun(@ldivide, p, dp);       % likelihood ratio, (km)*N 
    pgdata = bsxfun(@times, lrdata, Gdata);     % policy gradient, (km)*N 
    pgmean = mean(pgdata, 2); 

    theta = theta + alpha*pgmean; 
    fprintf('norm(pg)/norm(theta) = %f\n', norm(pgmean)/(norm(theta)+1e-9)); 
    theta_traj(:,iter) = theta; 
    figure(12); hold on; plot(iter, J_traj(iter)/episode_num, ...
        'linestyle', 'none', 'marker', 'o'); 
end

% plot final policy trajectory 
sdata = reshape(sdata, [2 floor(T/2) episode_num]); 
Gdata = reshape(Gdata, [1 floor(T/2) episode_num]); 
figure; plot(1:floor(T/2), squeeze(sdata(1,:,:))); xlabel('step'); ylabel('angle'); 
figure; plot(1:floor(T/2), squeeze(sdata(2,:,:))); xlabel('step'); ylabel('angle speed'); 
figure; plot(1:floor(T/2), squeeze(Gdata(1,:,:))); xlabel('step'); ylabel('return'); 





function [straj, atraj, rtraj] = rolling_model(s0, T, A, dfunc, pfunc, rfunc)
% evolve system trajectory according to initial state, action set and
% policy function 
% input: initial state s0 [n*1]
%        evoluation length T [1*1]
%        action set A {m*1}
%        dynamics function dfunc @1*1 
%        policy function pfunc @1*1 
%        reward function rfunc @1*1 
% output: state trajectory straj [n*T]
%         action trajectory atraj [m*T]
%         reward trajectory rtraj [1*T]

% size
n = length(s0); 
m = length(A); 

% assign space 
straj = repmat(s0, 1, T); 
atraj = zeros(m,T); 
rtraj = zeros(1,T); 

% repeat 
for t = 1:T
    s = straj(:,t);     % n*1 
    p = pfunc(s);       % m*1 
    a = randsample(m, 1, true, p);  % 1*1 
    r = rfunc(s, A{a});     % 1*1 
    sn = dfunc(s, A{a});    % n*1
    
    atraj(a,t) = 1; 
    rtraj(1,t) = r; 
    straj(:,t+1) = sn; 
end
straj = straj(:,1:T); 

end



function distribution = policy0(s,c,B,theta)
% policy distribution for given state
% input: state s [n*1]
%        center c [n*k]
%        bandwidth B [n*n*k]
%        policy params theta [km*1]
% output: distribution [m*1]

% size 
n = length(s); 
[~,k] = size(c); 
m = length(theta) / k; 

% all action identicator 
actions = eye(m);       % m*m 

aug_s = repmat(s, 1, m);      % n*m 
phis = basis_function(aug_s, actions, c, B);    % (km)*m
ys = phis'*theta;       % m*1 
distribution = exp(ys) / sum(exp(ys));  % m*1

% distribution = (distribution+0.1)/1.3; 

end 


function [p,dp] = policy(s,a,c,B,theta)
% probability and derivative for given state and action
% input: state s [n*1/N]
%        action a [m*1/N] 
%        center c [n*k]
%        bandwidth B [n*n*k]
%        policy params theta [km*1/N]
% output: probability p [1*N]
%         derivative dp [km*N]

% replicate theta or s/a
if size(theta,2) == 1
    theta = repmat(theta, 1, size(s,2));    % (km)*N
end
if size(s,2) == 1
    s = repmat(s, 1, size(theta,2));        % n*N
    a = repmat(a, 1, size(theta,2));        % m*N
end

% size 
[n, N] = size(s); 
[m, ~] = size(a); 
[~, k] = size(c); 

% phi for given action 
phi = basis_function(s, a, c, B);       % (km)*N
y = sum(bsxfun(@times, phi, theta), 1);     % 1*N
y_exp = exp(y);         % 1*N

% phi for all action 
actions = eye(m);       % m*m 
aug_actions = reshape(repmat(actions, 1, 1, N), [m m*N]);   % m*m*N -> m*(mN)
aug_s = reshape(repmat(s, m, 1), [n m*N]);      % (nm)*N -> n*(mN)
phi_all = basis_function(aug_s, aug_actions, c, B);     % (km)*(mN) 
phi_all = reshape(phi_all, [k*m m N]);      % (km)*m*N
y_all = sum(bsxfun(@times, phi_all, ...
    permute(theta, [1 3 2])), 1);    % 1*m*N
y_all = reshape(y_all, [m N]); 
y_all_exp = exp(y_all);         % m*N

% given (s,a) probability 
p = y_exp ./ sum(y_all_exp, 1);      % 1*N 

if nargout == 2
    z = bsxfun(@times, permute(y_all_exp, [3 1 2]), phi_all); % (km)*m*N
    z = bsxfun(@rdivide, squeeze(sum(z,2)), sum(y_all_exp,1));  % (km)*N
    dp = bsxfun(@times, p, phi-z);  % (km)*N
%     dp = dp/1.3; 
end

% p = (p+0.1)/1.3; 

end 


function check(fun, x)
% check if derivative is right 

[y, dy] = fun(x); 
fddy = finite_difference(fun, x, 1e-6); 
fprintf('norm x = %f, norm y = %f\n', norm(x), norm(y)); 
fprintf('norm (dy-fddy) = %f\n', norm(dy-fddy')); 

end


function J = finite_difference(fun, x, h)
% simple finite-difference derivatives
% assumes the function fun() is vectorized

if nargin < 3
    h = 2^-17;
end

[n, K]  = size(x);
H       = [-h*eye(n) h*eye(n)];
H       = permute(H, [1 3 2]);
X       = pp(x, H);
X       = reshape(X, n, K*(n+n));
Y       = fun(X);
m       = numel(Y)/(K*(n+n));
Y       = reshape(Y, m, K, n+n);
J       = pp(Y(:,:,n+1:end), -Y(:,:,1:n)) / (2*h);
J       = permute(J, [1 3 2]);

end

% utility functions: singleton-expanded addition and multiplication
function c = pp(a,b)
c = bsxfun(@plus,a,b);
end



function output = basis_function(s,a,c,B)
% basis function for RBF
% input: state s [n*N]
%        action a [m*N] 
%        center c [n*k]
%        bandwidth B [n*n*k]
% output: feature vector x [(km)*N]

% size 
[n, N] = size(s); 
[m, ~] = size(a); 
[~, k] = size(c); 

% first calculate basis for state 
d = bsxfun(@minus, permute(s, [1 3 2]), c);       % distance between state and center, n*k*N
dB = sum(bsxfun(@times, permute(d, [1 4 2 3]), B), 1);   % (n*1*k*N) * (n*n*k) sum@1-> (1*n*k*N)
dBd = sum(bsxfun(@times, dB, permute(d, [4 1 2 3])), 2);   % (1*n*k*N) * (1*n*k*N) sum@2-> (1*1*k*N), 
                    % (s-c)^T*B*(s-c)

phi = exp(-0.5 * squeeze(dBd));      % basis for state, k*N

% second determine action identicator 
% input action is represented by one-hot vector, i.e. j-th entry is 1,
% others are zeros, indicating aj action 
aug_phi = repmat(phi, m, 1);        % replicated state basis, (km)*N
aug_action = repmat(permute(a, [3 1 2]), k, 1, 1);      % replicated identicator, k*m*N
aug_action = reshape(aug_action, [m*k N]);      % (mk)*N

output = bsxfun(@times, aug_phi, aug_action); 

end

