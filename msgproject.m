function [ U, SS ] = msgproject(U,S,d,k,eps )
%FULLPROJ projects to the convex set of constraints sum(S)=1, S>=0, S<=1/k
%         using binary search. Note that this algorithm is not efficient.
%         For an efficient algorithm please refer to Arora et al 2013
%
%   U   the corresponding eigen-vectors of the (scaled) projection matrix M
%   S   the vector of eigen-values of the (scaled) projection matrix M
%   d   dimension of the ambient space
%   k   PCA dimension (desired dimension)
%   eps the maximum error that can be tolerated in projection

nz=S>0; S=S(nz); U=U(:,nz);
[S,srt]=sort(S,'descend'); U=U(:,srt);
flag = true;
ls=-d; rs=d;
while flag
    shf=(ls+rs)/2;
    SS=S+shf; SS(SS<0)=0; SS(SS>1/k)=1/k;
%     1-sum(SS)
    if sum(SS)-1>eps
        rs=shf;
    elseif 1-sum(SS)>eps
        ls=shf;
    else
        flag=false;
    end
end
end
