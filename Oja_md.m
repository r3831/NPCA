function [ U ] = Oja_md( obs,q,U,eta,x_tilde )
%obs is the indexing set of the observed entries
%U is the dxk orthogonal matrix used for the projection
%x_tilde is the observed vector

%We run Oja's algorithm with unbiased gradients constructed from the
%vectors with missing entries x_tilde
    d=size(U,1);
    r=length(obs);
    alpha=1/q;
    x_hat=alpha*x_tilde;
    beta=(sqrt(r-r*q))/q;
    i_s=randi(r);
    z=zeros(d,1);
    z(obs(i_s))=beta*x_tilde(obs(i_s));
    [U,~] = qr(U + eta*(x_hat*(x_hat'*U) - z*(z'*U)),0);

end

