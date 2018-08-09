
%% Performs the "positive" Frobenius-MD update. The iterate M is:
%     M = U * diag( S ) * U'
%
% k - the dimension of the subspace which we seek
% U, S - "nontrivial" eigenvectors and eigenvalues of the iterate
% eta - the step size
% x - the sample vector
% eps - threshold for rank1update and msgproject
%%
function [U,S]=msg(k,U,S,eta,x,eps)

d=size(U,1);
[U,S]=rank1update(U,S,eta,x,eps);
% if(min(S)<0 || max(S)>1/k || abs(sum(S)-eps)>1)
%     disp('good!')
% end
[U,S]=msgproject(U,S,d,k,eps);
end
