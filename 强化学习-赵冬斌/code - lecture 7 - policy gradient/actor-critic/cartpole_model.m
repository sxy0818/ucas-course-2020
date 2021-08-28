function dx = cartpole_model(t,x,flag,u)

%parameters:
g=9.8;
length=0.5;
m_cart=1.0;
m_pend=0.1;

friction_cart=5.e-4;
friction_pend=2.e-6;
%friction_cart=0;
%friction_pend=0;

phi = 0;


%inputs:
force=u(1);

%states:
angle=x(1);
angular_vel=x(2);
distance=x(3);
vel=x(4);

%equations:
total_m=m_cart+m_pend;
momt_pend=m_pend*length;

hforce=momt_pend*(angular_vel^2)*sin(angle);
part_num=-force-hforce+friction_cart*sign(vel);

denom_ang_vel=length*(4/3-m_pend*((cos(angle))^2)/total_m);
num_ang_vel=g*sin(angle)*cos(phi)+cos(angle)*part_num/total_m-friction_pend*angular_vel/momt_pend;

dxdt1=angular_vel;
dxdt2=num_ang_vel/denom_ang_vel;

num_vel=force-total_m*g*sin(phi)+hforce-momt_pend*dxdt2*cos(angle)-friction_cart*sign(vel);

dxdt3=vel;
dxdt4=num_vel/total_m;

%output:
dx=[dxdt1;dxdt2;dxdt3;dxdt4];
