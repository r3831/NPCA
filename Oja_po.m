function [ U ] = Oja_po( obs,U,eta,x_tilde )
%obs is the indexing set of the observed entries
%U is the dxk orthogonal matrix used for the projection
%x_tilde is the observed vector

%We run Oja's algorithm with unbiased gradients constructed from the
%partially observed vectors x_tilde(obs)

    d=size(U,1);
    r=length(obs);
    alpha=sqrt( (d*(d-1))/(r*(r-1)) );
    x_hat=alpha*x_tilde;
    beta=sqrt((d*r-r^2)/(r-1));
    i_s=randi(r);
    z=zeros(d,1);
    z(obs(i_s))=beta*x_tilde(obs(i_s));
    [U,~] = qr(U + eta*(x_hat*(x_hat'*U) - z*(z'*U)),0);
    
end

