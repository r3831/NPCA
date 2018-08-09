addpath(genpath('../../NPCA'))

%% Init parameters
k=2; % desired rank
ob_frac=2; % observation fraction
numiters=5; % number of runs


eta=1; % for jw11
% eta=.0000001; % for mnist


methods = {'batch','msg','msg-po','msg-md','sgd','oja-po','oja-md','grouse'};


%% YOU CAN RUN THE EXPERIMENTS ON YOUR OWN DATASET BY LOADING IT HERE:
%% Arbitrary Dataset
%load('JW11.mat'); dataname='JW11'; tau=0; data=data.view1;
% load('mnist.mat'); dataname='MNIST'; tau=0;
% load('sMNIST.mat'); dataname='MNIST'; tau=0;

%% OR SAMPLE RANDOMLY JUST TO SEE HOW IT LOOKS:
%% Sample from some distribution
nn=1e4;
X=rand(100,nn);
[uu, ~, vv]=svd(X,'econ');
ss=2.^(0:-.1:-9.9);
X=uu*diag(ss)*vv'*sqrt(nn);
data.training=X(:,1:.8*nn);
data.tuning=X(:,.8*nn+1:.9*nn);
data.testing=X(:,.9*nn+1:nn);
dataname='syn';
save ../data/syn.mat data

% d=32; N=6000; tau=1.8; dataname=sprintf('orthogonal_N=%d_d=%d_tau=%g',N,d,tau);
% [X,~]=simdist(d,3*N,tau);
% data=struct('training',X(:,1:N),'tuning',X(:,N+1:2*N),...
%     'testing',X(:,2*N+1:3*N));

[d,N]=size(data.training);
r=round(d/ob_frac); q=1/ob_frac; % let r and q be consistent
mu=mean(data.training,2);
data.training=data.training-repmat(mu,1,N);
data.tuning=data.tuning-repmat(mu,1,size(data.tuning,2));
data.testing=data.testing-repmat(mu,1,size(data.testing,2));

stochPCA(dataname,methods,data,k,numiters,eta,r,q);
methods={'batch','msg','msg-po','msg-md','sgd','oja-po','oja-md','grouse'};
plotobjV(dataname,methods,k,N,numiters,eta,d,ob_frac,methods);
