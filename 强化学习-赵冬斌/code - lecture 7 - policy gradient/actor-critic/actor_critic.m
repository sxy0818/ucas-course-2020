% this code is a complete simulation program used to generate data 
% for the cart-pole balancing problem
% reported in "On-line learning control by association and reinforcement" 
% co-authored by Jennie Si and Yu-Tsung Wang (IEEE Transactions on Neural Networks,
% 12(2):264-276, 2001.)
% Yu-Tsung Wang, Russell Enns, and Lei Yang have made major contributions to this code.
% This version is created and mained by Lei Yang and Li Guo.
% This code is a propterty of Prof. Jennie Si's research group.
% October, 2002.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
% flops(0);
%hold off

%%==================================================================
%% Initialize action network type: 
%% linear network ---- ANN=0
%% nonlinear network with one hidden layer -------- ANN=1.
ANN = 1;	   

%%==================================================================
%% Initialize critic network type: 
%% linear network ---- CNN=0
%% nonlinear network with one hidden layer -------- CNN=1.
CNN = 1;	   

%%==================================================================
%% Control type: 
%% Binary control ------ CTL_TYPE=0
%% Analylog control ---- CTL_TYPE=1.
CTL_TYPE = 0;	   

%%==================================================================
%% Plot Result: 
%% Don't plot intermediate result ------ PLOT_TYPE=0
%% Plot intermediate result       ------ PLOT_TYPE=1
PLOT_TYPE = 1;	   


%%=====================================================
%% Initialize parameters
Rad2Ang=180/pi;
Ang2Rad=pi/180;

alpha=0.95;	   %discount factor
Initlc=0.3;		%initial learning rate for critic
Initla=0.3;		%initial learning rate for action
Ta = 0.005;
Tc = 0.05;

FailDeg = 12.0;
FailTheta = (FailDeg*Ang2Rad);
Boundary =  2.4;

Mag = 10;   % Control force magnitude

WA_Inputs = 4;
WC_Inputs = 5;

INIT_THETA = (0.0*Ang2Rad);      % (rad)   was 2.0
INIT_THETADOT = (0.0*Ang2Rad);   % (rad/s) was 2.0
INIT_X = 0.0; 	                % (m)     was 0.25
INIT_XDOT = 0.0; 	            % (m/s)   was 0.25 


inistate = [INIT_THETA INIT_THETADOT INIT_X INIT_XDOT];

NF_Theta = FailTheta;
NF_ThetaDot = (120.0*Ang2Rad);
NF_x = Boundary;
NF_xDot = 1.5;

NF = [NF_Theta NF_ThetaDot NF_x NF_xDot];


Ncrit = 50;
Nact  = 100;

tstep=0.02;

% Maximum time assumed the pole will balanced forever,the unit is seconds
MaxTime = 120;      %(s)   was 120 seconds
Tit=MaxTime/tstep;

%Desired number of runs
MaxRun=5;
%Desired number of trials
MaxTr=1000;

%Number of nodes in hidden layer
N_Hidden = 6;

%Objective value of the cost function
Uc=0;

%%=====================================================
%% initialization
%%=====================================================
failure=0;
trbar = 0;
sucb = 0;
failRun =0;
newSt = inistate;
inputs = newSt./NF;		%initial state

ExpHist=[];

%Temp varible
crit_cyc=[];
action_cyc=[];
lc_hist=[];
la_hist=[];

lfts_history = 6;
max_stuck_count = 5;
max_lfts_diff = 5;
trial_state=zeros(lfts_history,2);

TRIAL=[];

for runs = 1:MaxRun,
    
    %%===========================================================%%
    %% initialize weights for crit network and action network.   %%
	%%===========================================================%%
    switch CNN
    case 0
        wc=(rand(WC_Inputs,1)-0.5)*2;
    case 1
        wc1=(rand(WC_Inputs,N_Hidden)-0.5)*2;
        wc2=(rand(N_Hidden,1)-0.5)*2;
    end
   
    switch ANN
    case 0
        wa=(rand(WA_Inputs,1)-0.5)*2;
    case 1
        wa1=(rand(WA_Inputs,N_Hidden)-0.5)*2;
        wa2=(rand(N_Hidden,1)-0.5)*2;
        delta_wa1 = rand(1,N_Hidden);
        delta_wa2 = rand(N_Hidden,1);   
    end
   
    switch CNN
    case 0
        wc1hist = wc';    
    case 1
        wc1hist = wc1(1,1);    
    end
   
    switch ANN
    case 0
        wa1hist= wa';    
    case 1
        wa1hist = wa1(1,1);    
    end
   
    reinfhist=0;
   
    ehist = 0.0;
   
    delWaHist=[];
    delta_wa=[];
   
      
    lfts=1;
    lc = Initlc;
    la = Initla;
    cyc = 0;
    counter=0;
 
    switch ANN
    case 0
        %noise=randn/500;
        va=inputs*wa;
        newAction = (1 - exp(-va))/(1 + exp(-va)); %+ noise;

    case 1
        ha = inputs*wa1;
        g = (1 - exp(-ha))./(1 + exp(-ha));
        va = g*wa2;
        newAction = (1 - exp(-va))./(1 + exp(-va));
    end
   
    switch CNN
    case 0
       inp=[inputs newAction];
       J=inp*wc;
    case 1
       inp=[inputs newAction];
       qc=inp*wc1;
       p = (1 - exp(-qc))./(1 + exp(-qc));
       J=p*wc2;
    end
   
    xhist = newSt;
    if (newAction >= 0)
        uhist = 1;
    else
        uhist = -1;
    end
    Jhist = J;
         
    Jprev = J;
   
    for trial=1:MaxTr,
        found_fail=0;
        count=0;
        failure=0;
        failReason=0;
        lfts = 1;
        newSt = inistate;
        inputs = newSt./NF;
        lc = Initlc;
        la = Initla;
      
        switch ANN
        case 0
            va=inputs*wa;
            newAction = (1 - exp(-va))/(1 + exp(-va)); %+ noise;

        case 1
            ha = inputs*wa1;
            g = (1 - exp(-ha))./(1 + exp(-ha));
            va = g*wa2;
            newAction = (1 - exp(-va))./(1 + exp(-va));
        end
      
        switch CNN
        case 0
            inp=[inputs newAction];
            J=inp*wc;
        case 1
            inp=[inputs newAction];
            qc=inp*wc1;
            p = (1 - exp(-qc))./(1 + exp(-qc));
            J=p*wc2;
        end

        
        Jprev = J;

      
        while(lfts<Tit),
            counter=counter+1;
            if (rem(lfts,500)==0),
                disp(['It is ' int2str(lfts) ' time steps now......']);
            end
         
            % Control selection
            if (CTL_TYPE == 0)
                if (newAction >= 0)
                    sgnf = 1;
                else
                    sgnf = -1;
                end
                u = Mag*sgnf;		%bang-bang control
            else
                u=Mag*newAction;
            end

		    %Plug in the model
            [T,Xf]=ode45('cartpole_model',[0 tstep],newSt,[],u);
	        a=size(Xf);
	        newSt=Xf(a(1),:);
        
            inputs=newSt./NF;	%input normalization   
        
            switch ANN
            case 0
                va=inputs*wa;
                newAction = (1 - exp(-va))/(1 + exp(-va));
            case 1
                ha = inputs*wa1;
                g = (1 - exp(-ha))./(1 + exp(-ha));
                va = g*wa2;
                newAction = (1 - exp(-va))./(1 + exp(-va));
            end
   
            %calculate new J    
            switch CNN
            case 0
                inp=[inputs newAction];
                J=inp*wc;
            case 1
                inp=[inputs newAction];
                qc=inp*wc1;
                p = (1 - exp(-qc))./(1 + exp(-qc));
                J=p*wc2;
            end
    
                   
            xhist=[xhist; newSt];
            Jhist=[Jhist; J];
            uhist=[uhist; u];

		    %%===========================================================%%
		    %% reinforce can be ether binary or continuous               %%
		    %%===========================================================%%
         
            if (abs(newSt(1)) > FailTheta)
                reinf = -1;
                failure = 1;
                failReason = 1;
            elseif (abs(newSt(3)) > Boundary)
                reinf = -1;
                failure = 1;
                failReason = 2;
            else
                reinf = 0;
            end
            
            reinfhist=[reinfhist; reinf];
    
    	    %%================================%%
    	    %% learning rate update scheme    %%
    	    %%================================%%
           
            if (rem(lfts,5)==0)
                lc = lc - 0.05;
                la = la - 0.05;
            end
          
      	    if (lc<0.01)
                lc=0.005;
            end
          
            if (la<0.01)
                la=0.005;
            end
          
          
            %%================================================%%
            %% internal weights updating cycles for critnet   %%
            %%================================================%%
                   
            cyc = 0;
            ecrit = alpha*J-(Jprev-reinf);
		    Ec = 0.5 * ecrit^2;
            ehist=[ehist;Ec];

            while (Ec>Tc & cyc<=Ncrit),
                %linear
                switch CNN
                %linear critic network
                case 0
                    gradJwc = [inputs'; newAction];
                    wc = wc - lc*alpha*ecrit*gradJwc;
                    inp=[inputs newAction];
                    J=inp*wc;
                %nonlinear critic network
                case 1                 
                    gradEcJ=alpha*ecrit;
    				%----for the first layer(input to hidden layer)-----------
                    gradqwc1 = [inputs'; newAction];
                    for i=1:N_Hidden,
                        gradJp = wc2(i);
                        gradpq = 0.5*(1-p(i)^2);
                        wc1(:,i) = wc1(:,i) - lc*gradEcJ*gradJp*gradpq*gradqwc1;
                    end
                
                    %----for the second layer(hidden layer to output)-----------
                    gradJwc2=p';
                    wc2 = wc2- lc*gradEcJ*gradJwc2;
                 
                    %----compute new  J----
                    inp=[inputs newAction];
                    qc=inp*wc1;
                    p = (1 - exp(-qc))./(1 + exp(-qc));
                    J=p*wc2;

                end
             
                cyc = cyc +1;
                ecrit = alpha*J-(Jprev-reinf);
                Ec = 0.5 * ecrit^2;
            end % end of "while (Ec>0.05 & cyc<=Ncrit)"
          
            crit_cyc=[crit_cyc cyc];
                
            %normalization weights for critical network
            switch CNN
            case 0
                if (max(max(abs(wc)))>1.5)
                    wc=wc/max(max(abs(wc)));
                end
            case 1
                if (max(max(abs(wc1)))>1.5)
                    wc1=wc1/max(max(abs(wc1)));
                end
                if max(max(abs(wc2)))>1.5
                    wc2=wc2/max(max(abs(wc2)));
                end
            end
      
            %%=============================================%%
            %% internal weights updating cycles for actnet %%
            %%=============================================%%                
            cyc = 0;
            
            eact = J - Uc;
            Ea = 0.5*eact^2;
            while (Ea>Ta & cyc<=Nact),
                graduv = 0.5*(1-newAction^2);
              
                switch ANN
                %linear action network
                case 0
                    %linear critic network
                    gradJu = wc(WC_Inputs);
                    delta_wa=-la*(J-0)*gradJu*graduv*inputs';
                    wa = wa +delta_wa;
                   
                    va=inputs*wa;
                    newAction = (1 - exp(-va))/(1 + exp(-va));
  
                    inp=[inputs newAction];
                    J=inp*wc;

                    %nonlinear action network
                case 1
                    gradEaJ = eact;
                    switch CNN
                    case 0
                        gradJu = wc(WC_Inputs);
                    case 1
                        gradJu = 0;
                        for i=1:N_Hidden,
                            gradJu = gradJu + wc2(i)*0.5*(1-p(i)^2)*wc1(WC_Inputs,i);
                        end
                    end %end of "switch(CNN)"
                
                    %----for the first layer(input to hidden layer)-----------
                    for (i=1:N_Hidden),
                        gradvg = wa2(i);
                        gradgh = 0.5*(1-g(i)^2);
                        gradhwa1 = inputs';
                        delta_wa1=-la*gradEaJ*gradJu*graduv*gradvg*gradgh*gradhwa1;
                        wa1(:,i) = wa1(:,i) + delta_wa1;
                    end
             
                    %----for the second layer(hidden layer to output)-----------
                    gradvwa2 = g';
                    delta_wa2=-la*gradEaJ*gradJu*graduv*gradvwa2;
                    wa2 = wa2 + delta_wa2;
                  
                    %----compute new J and newAction------- 
                    ha = inputs*wa1;
                    g = (1 - exp(-ha))./(1 + exp(-ha));
                    va = g*wa2;
                    newAction = (1 - exp(-va))./(1 + exp(-va));

                    switch CNN
                    case 0
                        inp=[inputs newAction];
                        J=inp*wc;

                    case 1
                        inp=[inputs newAction];
                        qc=inp*wc1;
                        p = (1 - exp(-qc))./(1 + exp(-qc));
                        J=p*wc2;

                    end %end of "switch(CNN)"
                end %end of "switch(ANN)"
                
                cyc = cyc+1;
                eact = J - Uc;
                Ea = 0.5*eact^2;
                
            end %end of "while (Ea>Ta & cyc<=Nact)"
            
            action_cyc=[action_cyc cyc];
           
            if ~failure
                Jprev=J;
                lc_hist=[lc_hist lc];
                la_hist=[la_hist la];
                switch ANN
                case 0
                    delWaHist=[delWaHist;delta_wa];
                case 1
                    delWaHist=[delWaHist;delta_wa1(1,1)];
                end
            else
                %another trial
                %Exit the while loop.
                break;
            end
    
            switch CNN
            case 0
                wc1hist=[wc1hist; wc'];    
            case 1
                wc1hist=[wc1hist; wc1(1,1)];    
            end
           
            switch ANN
            case 0
                wa1hist=[wa1hist; wa'];    
            case 1
                wa1hist=[wa1hist; wa1(1,1)];    
            end
           
            lfts=lfts+1;
        end  %end of "while(lfts<Tit)"
      
        if (PLOT_TYPE == 1)
            figure(1);
      
            subplot(5,1,1);
            plot(xhist(:,1)*Rad2Ang);
            title('Angle(Degree)');
            grid on;
      
            subplot(5,1,2);
            plot(xhist(:,2));
            title('Omega (Rad/Sec)');
            grid on;
      
            subplot(5,1,3);
            plot(xhist(:,3));
            title('Distance(Meter)');
            grid on;
      
            subplot(5,1,4);

            plot(xhist(:,4));
            title('Velocity(Meter/Sec)');
            grid on;
      
            subplot(5,1,5);
            plot(uhist)
            title('Control Force (Newton)');
            grid on;
      
            xlabel('time steps');

            figure(2)
        
            subplot(3,2,1);
            plot(Jhist)
            title('predicted J')
            grid on;
            
            subplot(3,2,2);
            plot(reinfhist)
            title('Trajectory of reinforcement values');
            grid on;
        
            subplot(3,2,3);
            plot(wc1hist)
            title('1st component of Weight vector for Critic');
            xlabel('time steps')
            grid on;
        
            subplot(3,2,4);
            plot(wa1hist)
            title('1st component of Weight vector for Action');
            grid on;
        
            subplot(3,2,5);
            plot(ehist)
            title('square error');
            grid on;
        
            subplot(3,2,6);
            plot(delWaHist)
            title('Trajectory of delta Wa');
            xlabel('time steps');
            grid on;
        
            figure(3)
            
            subplot(4,1,1);
            plot(crit_cyc);
            title('Critic cycle');
            grid on;
        
            subplot(4,1,2);
            plot(action_cyc);
            title('Action cycle');  
            grid on;
        
            subplot(4,1,3);
            plot(lc_hist);
            title('Critic learning rate');
            grid on;
        
            subplot(4,1,4);
            plot(la_hist);
            title('Action learning rate');  
            xlabel('time steps');
            grid on;
            
        end
    %   crit_cyc=[];action_cyc=[];
        la_hist=[];lc_hist=[];
          
        msgstr1=['Trial # ' int2str(trial) ' has  ' int2str(lfts) ' time steps.'];
   	    msgstr21=['Trial # ' int2str(trial) ' has successfully balanced for at least '];
   	    msgstr22=[msgstr21 int2str(lfts) ' time steps '];

        if ~failure
            disp(msgstr22);
            trbar = trbar + trial;
            sucb = sucb + 1;
            ExpHist=[ExpHist; runs trial failReason];
   		    break;
        else 
            disp(msgstr1);
        end
        
        fprintf('press any button to continue...\n'); 
        pause; 
        if (rem(trial,100)==0 & trial~=MaxTr)
            msgstr3=['Press any key to continue...'];
            disp(msgstr3);
        end
        
        if (trial~=MaxTr)
            reinfhist=0;
            ehist=[];
            wc1hist=[];
            wa1hist=[];
            Jhist=Jprev;
            uhist=u;
        end
      
  %======================================================================      
  % weights in and compare to previously stored old values
  %======================================================================
   if trial~=MaxTr
           for s=1:lfts_history-1
               trial_state(s,:) = trial_state(s+1,:);
           end
           
           trial_state(lfts_history,:) = [lfts,(~failure)];
           
           for t=1:lfts_history
               
               if trial_state(t,2)==1
                  found_fail=1;
              end
           end
           if found_fail==0
              for i=1:lfts_history-1
                  if (abs(trial_state(i,1)-trial_state(i+1,1))<=max_lfts_diff)
                      count=count+1;
                  end
              end
          end
          if count==max_stuck_count
         %%===========================================================%%
         %% initialize weights for crit network and action network.   %%
	     %%===========================================================%%
             switch CNN
             case 0
                 wc=(rand(WC_Inputs,1)-0.5)*2;
             case 1
                 wc1=(rand(WC_Inputs,N_Hidden)-0.5)*2;
                 wc2=(rand(N_Hidden,1)-0.5)*2;
             end
   
             switch ANN
             case 0
                 wa=(rand(WA_Inputs,1)-0.5)*2;
             case 1
                 wa1=(rand(WA_Inputs,N_Hidden)-0.5)*2;
                 wa2=(rand(N_Hidden,1)-0.5)*2;
                 delta_wa1 = rand(1,N_Hidden);
                 delta_wa2 = rand(N_Hidden,1);   
             end
 
         
          end
      end
      ExpHist=[ExpHist; runs trial failReason];
        
    end % end of for trials
    
       	    
fprintf('\nRun %d successful at trial #%d\n\n',runs,trial);
TRIAL=[TRIAL trial];
    
end %end of for runs

if failure
    failRun = failRun + 1;
end
Percent_run=sucb/runs*100;
avgtr = trbar/sucb;
stand_devia = std(TRIAL);

fprintf('\nTotal number of runs is %d\n\n', MaxRun);
fprintf('Number of successful runs is %d\n\n', sucb);
fprintf('Percentage of successful runs is %3.1f%%\n\n',Percent_run);
fprintf('Average number of trials to success is %4.1f\n\n',avgtr);

%output 
fid = fopen('result.txt', 'w');
fprintf(fid, 'Result Description:\n');
fprintf(fid, '  Initial State:\n');
fprintf(fid, '         Initial angle of pole          %5.2f\n',INIT_THETA);
fprintf(fid, '         Initial angular speed of pole  %5.2f\n',INIT_THETADOT);
fprintf(fid, '         Initial position of cart       %5.2f\n',INIT_X);
fprintf(fid, '         Initial speed of cart          %5.2f\n',INIT_XDOT);

fprintf(fid, '  Failure Criterion:\n');
fprintf(fid, '         Failure angle     %3.1f (degree)\n',FailDeg);
fprintf(fid, '         Out of bound      %3.1f (meter)\n',Boundary);

fprintf(fid, '  Training Parameters:\n');
fprintf(fid, '         Total number of runs                 %d \n',MaxRun);
fprintf(fid, '         Maximum trial numbers in one run     %d \n',MaxTr);
fprintf(fid, '         Time in one trial                    %d (second)\n',MaxTime);
fprintf(fid, '         Step length                          %4.3f (second)\n',tstep);

fprintf(fid, '  Running Pocess:\n');
fprintf(fid, '         Run      Trial\n');

for i=1:MaxRun
    fprintf(fid, '          %d        %d\n',i,TRIAL(i));
end

fprintf(fid, '  Running Result:\n');
fprintf(fid, '         Total number of runs                            %d\n', MaxRun);
fprintf(fid, '         Number of successful runs                       %d\n', sucb);
fprintf(fid, '         Percentage of successful runs                   %3.1f%%\n',Percent_run);
fprintf(fid, '         Average number of trials to success            %4.1f\n',avgtr);
fprintf(fid, '         Standard deviation of trial number to success   %3.1f\n',stand_devia);



fclose(fid);

