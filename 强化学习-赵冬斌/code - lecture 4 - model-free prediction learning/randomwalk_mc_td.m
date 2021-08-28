%% reinforcement learning course
% model-free prediction learning
% random walk evaluate a value with MC/TD method 

close all; clear; 

% state transition matrix [P(i,j)]
%       i: state {A,B,C,D,E}
%       j: successor state {L,A,B,C,D,E,R}
%     L,  A,  B,  C,  D,  E,  R
P = [ 1,  0,  0,  0,  0,  0,  0;        % L
     0.5, 0, 0.5, 0,  0,  0,  0;        % A
      0, 0.5, 0, 0.5, 0,  0,  0;        % B
      0,  0, 0.5, 0, 0.5, 0,  0;        % C
      0,  0,  0, 0.5, 0, 0.5, 0;        % D
      0,  0,  0,  0, 0.5, 0, 0.5;       % E
      0,  0,  0,  0,  0,  0,  1];       % R
  
% reward vector 
R = [0,0,0,0,0,0,1]; 

% discount factor 
gamma = 1; 

% true value 
Vtrue = [1/6, 2/6, 3/6, 4/6, 5/6]'; 
figure(1); plot(1:5,Vtrue,'marker','.'); xlabel('State'); ylabel('Estimated value'); 
xticks(1:5); xticklabels({'A','B','C','D','E'}); 

%% first-visit Monte-Carlo method and temporal-difference learning 
% sample times 
M = 100; 

% learning rate array 
alpha_mc_set = [0.01,0.02,0.03,0.04]; 
alpha_td_set = [0.05,0.1,0.15]; 

% value hist 
VMChist_set = zeros(5,M,length(alpha_mc_set)); 
VTDhist_set = zeros(5,M,length(alpha_td_set)); 

% initialize value 
Vmc0 = 0.5*ones(7,1); 
Vmc_set = repmat(Vmc0,1,1,length(alpha_mc_set)); 
Vtd0 = 0.5*ones(7,1); 
Vtd_set = repmat(Vtd0,1,1,length(alpha_td_set)); 

% simulate episode
for k = 1:M
    s = 4;          % C position 
    shist = s;
    rhist = R(s); 
    while s~=1 && s~=7
        s = randsrc(1,1,[1:7; P(s,:)]); 
        shist = [shist,s]; 
        rhist = [rhist,R(s)]; 
    end
    
    % MC method update 
    Ghist = cumsum(rhist,'reverse'); 
    for t = 1:length(alpha_mc_set)
        Vmc = Vmc_set(:,:,t); 
        alpha = alpha_mc_set(t); 
        % first time visit
        for i = 1:7
            I = find(shist==i,1,'first'); 
            if ~isempty(I)
                Vmc(i) = Vmc(i) + alpha*(Ghist(I)-Vmc(i)); 
            end
        end
%         % every time visit 
%         for i = 1:length(shist)
%             Vmc(shist(i)) = Vmc(shist(i)) + alpha*(Ghist(i)-Vmc(shist(i))); 
%         end
        Vmc_set(:,:,t) = Vmc; 
        VMChist_set(:,k,t) = Vmc(2:end-1); 
    end
    
    % TD update 
    for t = 1:length(alpha_td_set)
        Vtd = Vtd_set(:,:,t); 
        alpha = alpha_td_set(t); 
        for i = 1:length(shist)
            if i<length(shist)
                delta = rhist(i)+gamma*Vtd(shist(i+1))-Vtd(shist(i)); 
            else
                delta = rhist(i)-Vtd(shist(i)); 
            end
            Vtd(shist(i)) = Vtd(shist(i)) + alpha*delta; 
        end
        Vtd_set(:,:,t) = Vtd; 
        VTDhist_set(:,k,t) = Vtd(2:end-1); 
    end
end

% plot MC update value  
figure(1); hold on; plot(1:5,Vmc0(2:end-1),'marker','.'); 
figure(1); hold on; plot(1:5,VMChist_set(:,1,1),'marker','.'); 
figure(1); hold on; plot(1:5,VMChist_set(:,10,1),'marker','.'); 
figure(1); hold on; plot(1:5,VMChist_set(:,100,1),'marker','.'); 
legend('true','0','1','10','100'); 

% plot MC/TD error curve
figure(2); 
for t = 1:length(alpha_mc_set)
    plot(1:M,sqrt(sum((VMChist_set(:,:,t)-Vtrue).^2,1)/5)); hold on; 
end
for t = 1:length(alpha_td_set)
    plot(1:M,sqrt(sum((VTDhist_set(:,:,t)-Vtrue).^2,1)/5)); hold on; 
end
xlabel('Walks/Episodes'); ylabel('RMS eror, averaged over states');
legend('MC \alpha=0.01','MC \alpha=0.02','MC \alpha=0.03','MC \alpha=0.04',...
    'TD \alpha=0.05','TD \alpha=0.1','TD \alpha=0.15'); 
