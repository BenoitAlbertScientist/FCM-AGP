function [u,v,S,iter,fobj] = FCM_AOAPG(x,c,parameters)
% Fuzzy C-Means with Alternating Optimization and Accelerated Proximal
% Gradient
%    [u,v,S,iter,fobj] = FCM_AOAPG(x,c,parameters)
%
%             J_FCM  - tau_1 ln(det(S_j)) + tau_2 norm(S_j)_fro
% 
% INPUTS
%   x: input matrix (attributs x objects)
%   c: number of desired clusters
%   Optional : 
%       parameters.distance: 0=euclidean, 1=mahalanobis
%       parameters.init    : 0=random initialization of the center, 
%                            1=initialization with
%                              ADMMeuclidian distance=0,itmax=50,r=2.5.
%                            2=specific initialization (gives centers g)
%       parameters.g
%       parameters.tau1    : penaly parameter of ln(det(S_j))(default:10^-1)
%       parameters.tau2    : penaly parameter of norm(S_j)_fro(default:10^-3)
%       parameters.tol     : tolerance (default:10^-3)
%       parameters.itmax   : maximal number iterations maximal (default:1000)
%       parameters.
%           iprint         : 1=display general informations (default:1,else:0)
%           iprint_inside  : 1=display at each iteration (default:0)
%
%
% OUTPUTS
%   u: fuzzy partition  (clusters x objects)
%   v: centroids (attributs x clusters)
%   S: cell of cov-matrix (clusters (attributs x attributs ))
%   iter: numbers of iteration
%
%  --------------------------------------------------------------------------
%  Author : Benoit Albert
%  mail   : benoit.albert@uca.fr
%  date   : 10-01-2023
%  version: 2
%  --------------------------------------------------------------------------

% imensions
if nargin<2  
    error('FCM needs two arguments');
else
    n=size(x,2); nd=size(x,1);
end

% ------------------------ Check parameters -------------------------
if ~isfield(parameters,'init') parameters.init=1;end
if ~isfield(parameters,'distance') parameters.distance=0;end
if ~isfield(parameters,'tau1') parameters.tau1=10^-1;end
if ~isfield(parameters,'tau2') parameters.tau2=10^-3;end
if ~isfield(parameters,'tol') parameters.tol=10^-3;end
if ~isfield(parameters,'itmax') parameters.itmax=1000;end
if ~isfield(parameters,'iprint') parameters.iprint=1;end
if ~isfield(parameters,'iprint_inside') parameters.iprint_inside=0;end

if (parameters.init~=0 && parameters.init~=1 && parameters.init~=2) parameters.init=1;end
if (parameters.distance~=0 && parameters.distance~=1) parameters.distance=0;end
if parameters.itmax<1 parameters.itmax=1000;end
if (parameters.iprint~=0 && parameters.iprint~=1) parameters.iprint=1;end
if (parameters.iprint_inside~=0 && parameters.iprint_inside~=1) parameters.iprint_inside=0;end

init=parameters.init;
distance=parameters.distance;
tau1=parameters.tau1;
tau2=parameters.tau2;
tol=parameters.tol;
itmax=parameters.itmax;
iprint=parameters.iprint;
iprint_inside=parameters.iprint_inside;

if iprint == 1
    fprintf('*******************************************\n');
    fprintf('\t Fuzzy C-means with AO and Nesterov \n');
    fprintf('-------------------------------------------\n');
    fprintf('Number of objects  = %5i\n',n);
    fprintf('Number of clusters = %5i\n',c);
end

% ---------------------- Initialization ----------------------
l_max = 2 * sqrt(2);
mu_min = 1 / sqrt(l_max);
delta = 1/(tau1 * nd * l_max + 2*tau2);
eps0=10^-10; %avoid fuzzy covariance to be null 


if init == 0 
% Random
    % Mass
    u=rand(n,c); su=sum(u,2);
    u=u./su(:,ones(1,c));

    % Center of mass
    u2=u.^2; su=sum(u2); v=x*u2;
    v=v./su(ones(nd,1),:);
    if iprint == 1;fprintf('Initialization : Random\n');end;
    
elseif init == 1
% Initialization with Euclidean distance    
    parameters_euc.init = 0;
    parameters_euc.distance = 0;
    parameters_euc.r = 2.5;
    parameters_euc.itmax = 50;
    parameters_euc.iprint = 0;
    [u_eu,v_eu,S_eu,iter_eu] = FCM_ADMM(x,c,parameters_euc);
    u = u_eu; v = v_eu; S = S_eu;
    if iprint == 1;fprintf('Initialization : Euclidan[iter=%2i  (max. 50)]\n',iter_eu);end;
elseif init == 2
% Specific initialization
    v = parameters.g;
    for i=1:n
        su=0;
        for k=1:c
            dd=(x(:,i)-v(:,k))'*(x(:,i)-v(:,k));
            su=su+1/dd;
        end
        for j=1:c
            dd=(x(:,i)-v(:,j))'*(x(:,i)-v(:,j));
            u(i,j)=max(0,1/dd/su);
        end
    end
    if iprint == 1;fprintf('Initialization : Specific\n');end;
        
end

% Inv- Var-Covariance
eps0 = 10^-8;
if distance == 0
    for j=1:c
        S{j}=eye(nd);
    end 
else 
    ux=u(:)';  ic=[1:c]'; ic=kron(eye(c),ones(1,n))'*ic;
    d=repmat(x,1,c)-v(:,ic');
    p=ux(ones(nd,1),:).*d;
    for j=1:c
        Sj=eps0*eye(nd); j1=(j-1)*n+1; j2=(j-1)*n+n;
        S{j}=p(:,j1:j2)*p(:,j1:j2)'+eps0*eye(nd);
        scal=(det(S{j}))^(1/nd); 
        S{j}=scal*inv(Sj);
    end 
end


% ------------------------- Iterations ----------------------------------
err=1;
iter=0;

while (iter < itmax && err>tol)
    iter=iter+1;
    uu=u; vv=v; SS=S;
    
    % Compute centers : V
    su=sum(u.*u,1);
    for j=1:c
        v(:,j)=zeros(nd,1);
        for i=1:n
            v(:,j)=v(:,j)+u(i,j)*u(i,j)*x(:,i);
        end
        v(:,j)=v(:,j)/su(j);
    end
    
    % Compute distance matrix : S
    if distance == 1
        for j=1:c
            S{j}=real(APG(j,x,v,u,mu_min,delta,tau1,tau2,iprint_inside));
        end   
    end

    % Compute partition :  U
    for i=1:n
        su=0;
        for k=1:c
            dd=(x(:,i)-v(:,k))'*S{k}*(x(:,i)-v(:,k))+eps0;
            su=su+1/dd;
        end
        for j=1:c
            dd=(x(:,i)-v(:,j))'*S{j}*(x(:,i)-v(:,j))+eps0;
            u(i,j)=max(0,1/dd/su);
        end
    end
    
    % Compute errors (stopping criterion)
    ndu=sum(sum((u-uu).^2));   nu=sum(sum(u.^2));
    ndv=sum(sum((v-vv).^2));   nv=sum(sum(v.^2));
     
    ndS=0; nS=0;
    for j=1:c
        nS=nS+norm(S{j},'fro')^2;
        ndS=ndS+norm(S{j}-SS{j},'fro')^2;
 
    end
    errd=[ndu;ndv;ndS];
    errn=[nu;nv;nS];
    
    err=sqrt(sum(errd)/sum(errn));

    if (iprint_inside == 1)    
        %Objectif function
        fobj = 0;
        for j=1:c
           Sj = S{j};
           for i=1:n
              pij = u(i,j)*(x(:,i)-v(:,j));
              fobj = fobj + pij'*Sj*pij;
           end
        end
       fprintf(">>iter=%i | err=%1.6e\n | J_FCM=%15.8e\n",iter,err,fobj);
    end
 
    
end

%Function
fobj = 0;
for j=1:c
   Sj = S{j};
   fobj = fobj - tau1*log(det(Sj))+tau2*norm(Sj,'fro');
   for i=1:n
      pij = u(i,j)*(x(:,i)-v(:,j));
      fobj = fobj + pij'*Sj*pij;
   end
end

if (iprint == 1)   
    fprintf('-------------------------------------------\n');
    fprintf("Objectif function F(U,V,S) =%e\n",fobj);
    fprintf("[iter=%i (max. %i)| err=%1.6e]\n",iter,itmax,err);
    fprintf('*******************************************\n');
   
end

end

function [S] = APG(j,x,v,u,mu_min,delta,tau1,tau2,iprint_inside)
% Accelerated Proximal Gradient for the cluster j :
%        sum_{i=1}^n (x_i-v_j)' S_j (x_i-v_j)
%           - tau_1 ln(det(S_j)) + tau_2 norm(S_j)_fro
%    [S] = APG(j,x,v,u,mu_min,delta,tau1,tau2,iprint_inside)
% 
% INPUTS
%   j             : cluster's number 
%   x             : input matrix (attributs x objects)
%   u             : fuzzy partition  (clusters x objects)
%   v             : centroids (attributs x clusters)
%   mu_min        : minimal eig value 
%   delta         : inverse of the Lipschitzien constraint 
%   tau1,tau2     : coefficiants
%   iprint_inside : display (default:0)
%
% OUTPUTS
%   S: inv of cov-matrix (attributs x attributs)

    n = size(u,1);
    nd = size(v,1);

    % Initialisation -------------------------------
    STOP_Nest = 0;  
    err = 0; err_min = 1e-3;
    iter = 0; itermax = 500;

    tk=1;

    Cj=zeros(nd);
    for i=1:n
        Cj=Cj+u(i,j)*u(i,j)*(x(:,i)-v(:,j))*(x(:,i)-v(:,j))';
    end
    S = inv(Cj+eye(nd)*10^-8);
    Z = S;

    RESTART = 0; %Restart counter
    
    % Iterations -----------------------------------
    while STOP_Nest == 0 
      iter = iter +1;
      
      %Descente direction
      Grad = Cj - tau1*inv(Z) + 2*tau2*Z;

      %Sk   
      Sp = S;
      S = Z - delta *Grad;
      %Spectral decomposition
      [Q,D] = eig(S);
      %Projection
      S =  Q * max(D,mu_min*(eye(size(D))))* Q';
      
      %tk-restart test 
      if Need_Restert(j,x,v,u,S,Sp,tau1,tau2)
            RESTART = RESTART +1;
            t_prec = 1;
      else
            t_prec = tk;
            tk = (1+sqrt(1+4*tk^2))/2;
      end
      
      %Zk
      Zp = Z;
      Z = S + (t_prec-1)/tk*(S-Sp);
      
      %stop criterion
      errZZ = norm(Z-Zp,'fro')^2;
      errZ = norm(Z,'fro')^2;
      err = errZZ/errZ;
      STOP_Nest = err<err_min || iter>itermax;
      
    end
    
    if (iprint_inside == 1)   
        fprintf('[NESTEROV | restarts : %d - iter : %d ]\n',RESTART,iter);
    end
end

function [res] = Need_Restert(j,x,v,u,Sj,Sjp,tau1,tau2)
   %Test of restart : minization of the function ?
   
   n = size(u,1);
   f = - tau1*log(det(Sj))+tau2*norm(Sj,'fro');
   fp = - tau1*log(det(Sjp))+tau2*norm(Sjp,'fro');
   for i=1:n
      f = f + u(i,j)*u(i,j)*(x(:,i)-v(:,j))'*Sj*(x(:,i)-v(:,j));
      fp = fp + u(i,j)*u(i,j)*(x(:,i)-v(:,j))'*Sjp*(x(:,i)-v(:,j));
   end
   res = f>fp;
end

