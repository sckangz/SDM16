%noncovex low-rank model based Top-N recommender systems
%This is the code for paper "Top-N Recommendation with Novel Rank Approximation" by zhao kang, qiang cheng 
%on SIAM International Conference on Data Mining (SDM16) 

function [hr,arhr] =sdm(Trainn,test,test_zhong,tol,maxIter,lambda1,lambda2,mu,gamma)
[m,n]=size(Trainn);
X=Trainn;
K=X'*X;



Y1=zeros(n);
Y2=zeros(n);
Y3=zeros(n);

Z1=rand(n);
Z2=Z1;
Z3=Z1;
ww=ones(n,1);


for i=1:maxIter
W=inv(3*mu*eye(n)+K)*(mu*(Z1+Z2+Z3)+Y1+Y2+Y3+K);
    for j = 1:size(W,1)
        W(j,j) = 0;
    end
    
    D=W-Y1/mu;
Z1=max(abs(D)-lambda1/mu,0).*sign(D);
    
E=W-Y2/mu;
[ Z2,nw] = DC(E,mu/2/lambda2,ww,.1,3);

ww=nw;


Z3=max(W-Y3/mu,0);

Y1=Y1-mu*(W-Z1);
Y2=Y2-mu*(W-Z2);
Y3=Y3-mu*(W-Z3);

mu=mu*gamma;

   funval(i) = 1/2*norm(X*W-X,'fro')+lambda1*norm(W,1)+lambda2*sum(1-exp(-ww/.1));
  
if((i>1)&(abs(funval(i) - funval(i-1)) < funval(i-1) * tol))   
        break
    end
end


[hr,arhr] = cal_res(W,Trainn,test,test_zhong);
end



function [ X,T ] = DC(D,rho,T0,a,b)

[U,S,V] = svd(D,'econ');

for t = 1:10
 
    [ X,T1 ] = DCInner(S,rho,T0,a,b,U,V);
    err = norm(T1-T0,'fro')/norm(T0,'fro');
    if err < 1e-3
       break
    end
    T0 = T1;
end
T = T1;
end


function [ X,t ] = DCInner(S,rho,J,epislon,funcs,U,V)
lambda=1/2/rho;
% t = svd(J,'econ');
S0 = diag(S);
switch funcs
    case 1
grad=(1+epislon)*epislon./(epislon+J).^2;
    case 2
grad=1./(1+J.^2);
   case 3
grad=1/epislon*exp(-J/epislon);
    case 4
        grad=1./(1+J.^2);
end

t=max(S0-lambda*grad,0);
X=U*diag(t)*V';
end


function [hr,arhr] = cal_res(W,Trainn,test,test_zhong)
%% 
% Calculate HR corresponds to N = 5, 10, 15, 20, 25 and ARHR corresponds to
% N=5
%
%% The main program

    As = Trainn;
    zhong = zeros(1,5);
    po = 0;
    REC = As * W;
    hr = zeros(1,5);
    for i = 1:size(Trainn,1)
        value = REC(i,test{i});
        value1 = REC(i,test_zhong(i));
        position = length(find(value > value1)) + 1;
        for n = 5:5:25
            if((position <= n)&(value ~= min(value1)))
                zhong(n/5) = zhong(n/5) + 1;
                if n == 10
                    po = po + 1/position;
                end     
            end
        end
    end
    hr = zhong/size(Trainn,1);
    arhr = po/size(Trainn,1);
end
    
        