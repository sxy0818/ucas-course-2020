% reinforcement learning course
% model-free control learning
% windy gridworld with Sarsa(lambda)

close all; clear; 

% draw flag 
draw_flag = false; 

% discount factor 
gamma = 1; 

% learning rate 
alpha = 0.1;

% epsilon exploration rate 
epsilon = 0.1; 

% eligibility trace factor 
lambda = 0.9; 

% Q table [Q(k,i,j)]
%      k: action 
%      i: x axis 
%      j: y axis
Q = zeros(4,10,7); 

% episode number 
N = 170; 

% episode length hist
Lhist = zeros(1,N); 

% starting point 
s0 = [1,4]; 

% Sarsa(lambda) learning 
for t = 1:N
    E = zeros(4,10,7); 
    s = s0; 
    draw_gridworld(s,Q,draw_flag)
    a = epsilon_greedy(s,Q,epsilon); 
    terminal = false; 
    len = 1; 
    while ~terminal
        [sn,r,terminal] = windy_gridworld(s,a); 
        an = epsilon_greedy(sn,Q,epsilon); 
        delta = r+gamma*Q(an,sn(1),sn(2))-Q(a,s(1),s(2)); 
        E(a,s(1),s(2)) = E(a,s(1),s(2))+1; 
        Q = Q + alpha*delta*E; 
        E = gamma*lambda*E; 
        s = sn; 
        draw_gridworld(s,Q,draw_flag)
        a = an; 
        len = len+1; 
    end
    Lhist(t) = len; 
end

% plot episode length
figure(1); plot(1:N,Lhist); xlabel('episode'); ylabel('episode length'); 

% plot episode per time step 
figure(2); plot(cumsum(Lhist),1:N); xlabel('time steps'); ylabel('episode'); 

% final results
s = s0; 
draw_gridworld(s,Q,true)
a = epsilon_greedy(s,Q,epsilon); 
terminal = false; 
len = 1; 
while ~terminal
    [sn,r,terminal] = windy_gridworld(s,a); 
    an = epsilon_greedy(sn,Q,epsilon); 
    s = sn; 
    draw_gridworld(s,Q,true)
    a = an; 
    len = len+1; 
end





function [sn,r,terminal] = windy_gridworld(s,a)
% windy gridworld model
% input:
%     s=(x,y) state axis with left-bottom corner as origin
%     a={1,2,3,4} action corresponds to {left,right,up,down}
% output:
%     sn=(x,y) successor state 
%     r reward signal
%     terminal signal if reach goal 

% state add action 
sn = s; 
switch a
    case 1          % left
        sn(1) = sn(1)-1; 
    case 2          % right
        sn(1) = sn(1)+1; 
    case 3          % up 
        sn(2) = sn(2)+1; 
    case 4
        sn(2) = sn(2)-1; 
end

% windy effect 
if sn(1)>=4 && sn(1)<=9
    sn(2) = sn(2)+1; 
end
if sn(1)>=7 && sn(1)<=8
    sn(2) = sn(2)+1; 
end

% restrict state axis 
sn(1) = min(10,max(1,sn(1))); 
sn(2) = min(7,max(1,sn(2))); 

% reward 
r = -1; 

% terminal state 
if sn(1)==8 && sn(2)==4
    terminal = 1; 
else
    terminal = 0; 
end

end

function a = epsilon_greedy(s,Q,epsilon)
% epsilon-greedy policy 
% input: 
%     s=(x,y) state axis with left-bottom corner as origin
%     Q(k,i,j) value with action k, state (i,j)
%     epsilon exploration rate 
% output: 
%     a=[1,2,3,4] action 

if rand()>epsilon
    [C,I] = max(Q(:,s(1),s(2))); 
    a = I; 
else
    a = randi([1,4],1); 
end

end

function draw_gridworld(s,Q,flag)
% draw the gridworld with agent on the box 

if ~flag
    return; 
end 

% area 
figure(111); clf; 
% xlim([0,10]); 
% ylim([0,7]); 
box off; axis equal; 
hold on; 

% grid 
[X,Y] = meshgrid(0:10,0:7); 
plot(X,Y,'k'); 
plot(X',Y','k'); 

% plot Q values and agent on grid 
for i = 1:10
    for j = 1:7
        if i==s(1) && j==s(2)
            color = [1,0,0]; 
        else
            v = max(Q(:,i,j)); 
            color = [1,1,1]*(1+v/20)+[0,1,0]*(-v/20); 
            color(color<0) = 0; 
            color(color>1) = 1; 
        end
        rectangle('position',[i-1,j-1,1,1],'facecolor',color); 
        text(i-1,j-1+0.14,['L:' num2str(Q(1,i,j),3)],'fontsize',7); 
        text(i-1,j-1+0.38,['R:' num2str(Q(2,i,j),3)],'fontsize',7); 
        text(i-1,j-1+0.62,['U:' num2str(Q(3,i,j),3)],'fontsize',7); 
        text(i-1,j-1+0.86,['D:' num2str(Q(4,i,j),3)],'fontsize',7); 
    end
end

% start and goal point 
text(0.3,3.5,'S','fontsize',20); 
text(7+0.3,3.5,'G','fontsize',20); 

end


