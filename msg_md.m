%% Performs the "positive" Frobenius-MD update. The iterate M is:
%     M = U * diag( S ) * U'
%
% k - the dimension of the subspace which we seek
% U, S - "nontrivial" eigenvectors and eigenvalues of the iterate
% eta - the step size
% x - the sample vector
% eps - threshold for rank1update and msgproject
%%
function [U,S]=msg_md(obs,q,k,U,S,eta,x_tilde,eps)

d=size(U,1);
r=length(obs);
alpha=1/q;
x_hat=alpha*x_tilde;
[U,S]=rank1update(U,S,eta,x_hat,eps);
beta=(sqrt(r-r*q))/q;
i_s=randi(r);
z=zeros(d,1);
z(obs(i_s))=beta*x_tilde(obs(i_s));
[U,S]=rank1update(U,S,-eta,z,eps);

% beta=(sqrt(1-q))/q;
% for i=1:r
%     z=zeros(d,1);
%     z(obs(i))=beta*x_tilde(obs(i));
%     [U,S]=rank1update(U,S,-eta,z,eps);
% end
[U,S]=msgproject(U,S,d,k,eps);
end
