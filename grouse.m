function [ U_t ] = grouse( obs, U, eta, x_tilde )
%Implementation of Grouse as presented in Balzano's first paper
%Input: obs - indexing set of the observed entries of x
%U - previous orthogonal matrix tracking the subspace based on
%x_1,..x_{t-1}
%eta - step size ~ 1/t
%x_tilde - current observed vector from x_t
%Output: U_t - orthogonal matrix tracking the subspace based on x_1,..x_t

U_obs = zeros(size(U));
U_obs(obs,:) = U(obs,:);
w = U_obs\x_tilde;
norm_weights = norm(w);
r = x_tilde - U_obs*w;
norm_residual = norm(r);
sG = norm_residual*norm_weights;
U_t = U+((cos(sG*eta)-1)*U*(w/norm(U*w)) + sin(sG*eta)*(r/norm_residual))*(w'/norm_weights);
%just to be safe
%U_t = orth(U_t);
end

