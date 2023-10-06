%% Download data 
addpath 'Affichage&Index&Method'
addpath 'Data'
load iris_n

n=size(x,2);    %Number of objects
nd=size(x,1);   %Number of attributs
c=length(cl);   %Number of clusters

%% COMPARAISON AO-AGP vs AO
%Apply on FCM-GK model. 

parameters.init = 1; %Init 
parameters.distance = 1; %Mahalanobis distance
parameters.iprint = 1;
%AO-AGP
name_meth = 'AO-AGP'; rng('default'); %Rand init
parameters.tau1 = 10^-1; parameters.tau2 = 10^-3;
[u,v,S,iter,fobj] = FCM_AOAPG(x,c,parameters);
fprintf(" /// det(mu_min I) = %1.3e /// \n",sqrt(2*sqrt(2))^(-nd));
for i =1:c; fprintf("det(S_%d) = %1.3e\n",i,det(S{i})); end;
EVAL(x,u,v,S,HP,name_data,name_meth);

% ----- AO
name_meth = 'AO'; rng('default'); %Rand init
[u,v,S,iter,fobj] = FCM_AO(x,c,parameters);
EVAL(x,u,v,S,HP,name_data,name_meth);

%% Evaluation 
%Evaluation with ARI, PE, XB and XBMW.
%Print in 2D clustering.

function [] = EVAL(x,u,v,S,HP,name_data,name_meth)

    %ARI
    hp=Fuzzy2Hard(u);
    fprintf("ARI  = %.2f \n",ARI(HP,Fuzzy2Hard(u)));
    
    %PE
    fprintf("PE  = %1.2f \n",PE(u));
    
    %FS
    fprintf("FS  = %1.2f \n",FS(x',u));
    
    %XB
    parameters_XB.choice_index=0;
    xb=XB(x',u,v',parameters_XB);
    fprintf("XB  = %1.2f \n",xb);
    
    %XBMW
    parameters_XBMW.choice_index=1;
    parameters_XBMW.give_cov=1; %S is inverse of covariance matrix
    parameters_XBMW.matrix=S;
    xbmw=XB(x',u,v',parameters_XBMW);
    fprintf("XBMW  = %1.2f \n",xbmw);
    
    %DISPLAY
    titre = strcat(name_data,name_meth);
    DisplayClustering2D(x',v',hp,S,titre);

end

