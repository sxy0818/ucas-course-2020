% motor_gridQI.m
% value function approximation
% policy iteration for continuous state-action space with grid 
% discretization 
clear; close all; clc; 

% state space 
max_s1 = pi; 
min_s1 = -pi; 
max_s2 = 16*pi; 
min_s2 = -16*pi; 

% define a grid number
grid_num1 = 1000; 
grid_num2 = 1000; 

% cell boundary
cell_bound1 = min_s1:(max_s1-min_s1)/(grid_num1):max_s1; 
cell_bound2 = min_s2:(max_s2-min_s2)/(grid_num2):max_s2; 
cell_bound_set = {cell_bound1, cell_bound2}; 

% action space 
max_a = 10; 
min_a = 10; 

% finite action set 
act_set = {-10,0,10}; 

% cell center point
cell_centr1 = (cell_bound1(1:end-1)+cell_bound1(2:end))/2; 
cell_centr2 = (cell_bound2(1:end-1)+cell_bound2(2:end))/2; 
[X,Y] = meshgrid(cell_centr1,cell_centr2); 
cell_centr_set = [X(:)'; Y(:)'];        % [n*N]

% cell center + actions -> successor state and reward 
sample_s = repmat(cell_centr_set,1,length(act_set)); 
sample_a = reshape(repmat(cell2mat(act_set),size(cell_centr_set,2),1),1,[]); 
[sample_sn,sample_r] = motor_model(sample_s,sample_a); 

% subscript of sample state, action, and successor state 
sample_s_sub = locate(sample_s,cell_bound_set); 
sample_a_sub = reshape(repmat([1:length(act_set)],size(cell_centr_set,2),1),1,[]); 
sample_sn_sub = locate(sample_sn,cell_bound_set); 

% discount factor 
gamma = 0.95; 

% Q value arrayed in matrix [s1*s2*a]
Q0 = zeros(grid_num1,grid_num2,length(act_set)); 
Q = Q0; 

% locate sn with all actions [(n+m)*(Nm)]
sample_snA_sub = [repmat(sample_sn_sub,1,length(act_set)); 
    reshape(repmat([1:length(act_set)],size(sample_sn_sub,2),1),1,[])]; 
sample_snA_ind = sub2ind(size(Q),sample_snA_sub(1,:),...
    sample_snA_sub(2,:),sample_snA_sub(3,:)); 

% locate (s,a) [(n+m)*N]
sample_sa_sub = [sample_s_sub; 
    sample_a_sub]; 
sample_sa_ind = sub2ind(size(Q),sample_sa_sub(1,:),sample_sa_sub(2,:),...
    sample_sa_sub(3,:)); 

% record Q diff history
Qdiffhist = zeros(1,300); 

% grid Q iteration 
for k = 1:300
    Qold = Q; 
    
    % find the max Q for sn
    qsn = reshape(Q(sample_snA_ind),[],length(act_set))'; 
    maxqsn = max(qsn,[],1); 
    
    % update Q
    Q(sample_sa_ind) = sample_r + gamma*maxqsn; 
    
    Qdiff = max(abs(Q(:)-Qold(:))); 
    Qdiffhist(k) = Qdiff; 
    fprintf('%d-th iteration, Q diff=%.3f\n',k,Qdiff); 
    if Qdiff < 1e-2
        break;
    end
end

% plot Q diff curve 
figure; plot(1:k,Qdiffhist(1:k)); grid on; 
xlabel('iteration'); ylabel('Q diff'); 

% plot Q shape under a = 0
s1 = min_s1:(max_s1-min_s1)/100:max_s1; 
s2 = min_s2:(max_s2-min_s2)/100:max_s2; 
[S1,S2] = meshgrid(s1,s2); 
S = [S1(:)';S2(:)']; 
S_sub = locate(S,cell_bound_set); 
Sa0_sub = [S_sub; repmat(2,1,size(S_sub,2))]; 
Sa0_ind = sub2ind(size(Q),Sa0_sub(1,:),...
    Sa0_sub(2,:),Sa0_sub(3,:)); 
qa0 = Q(Sa0_ind); 
figure; mesh(S1,S2,reshape(qa0,size(S1))); 
xlabel('\alpha [rad]');
ylabel('\alpha'' [rad/s]');
zlabel('Q (\alpha,\alpha'',0)');
xlim([-pi pi]); ylim([-16*pi 16*pi]);

% plot policy 
SA_sub = [repmat(S_sub,1,length(act_set));
    reshape(repmat(1:length(act_set),size(S_sub,2),1),1,[])]; 
SA_ind = sub2ind(size(Q),SA_sub(1,:),...
    SA_sub(2,:),SA_sub(3,:)); 
q = reshape(Q(SA_ind),[],length(act_set))'; 
[C,I] = max(q,[],1); 
h = cell2mat(act_set(I)); 
figure; ph = pcolor(S1,S2,reshape(h,size(S1))); 
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

function sub = locate(s,bound_set)
% locate the subscript of given input under boundaries 
% input: state s [n*N]
%        boundary set bound_set {n*[1*M]}
% output: subscript location sub [n*N]

sub = zeros(size(s)); 
for i = 1:size(s,1)
    for j = 1:size(s,2)
        x = find(s(i,j)>=bound_set{i},1,'last'); 
        if ~isempty(x) 
            sub(i,j) = min(x,length(bound_set{i})-1); 
        else
            sub(i,j) = 1; 
        end
    end
end

end

