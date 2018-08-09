
%%STOCHPCA(dataname,methods,data,k,numiter,eta,r,q,itr_st) runs various
%  stochastic PCA algs in noisy settings
%  for more info see http://proceedings.mlr.press/v80/marinov18a.html
%  The inputs are as follows
%  dataname - currently only 'orthogonal' implemented
%  methods  - name of streaming methods to be executed 
%  data     - the data, a struct of three fields training, tuning, testing
%   k       - a positive integer, denotes the desired RANK
%   numiter - number of iterations
%   eta     - step size multiplier. The sequence of step size used by
%             various algorithms is eta/t or eta/sqrt(t), where t is the
%             iteration index.
%   r,q     - observation fractions, refer to the paper
%   itr_st  - start from iteration itr_st
%  The output containing the objective on the dev set,population objective, 
%  and the singular value decomposition (U,S,V) is written to a
%  MATFILE in ../page/profile/PCA/METHOD/DATANAME, where METHOD is one of
%  'batch','msg','msg-po','msg-md','sgd','oja-po','oja-md','grouse'. For 
%
%
function stochPCA(dataname,methods,data,k,numiter,eta,r,q,iter_start)



%% Default is starting at iter_start=1;
if(nargin<9)
    iter_start=1;
end

%% Default is full observation q = 1
if(nargin<8)
    q=1;
end


%% Default is full observation r = d
if(nargin<7)
    r=size(data.training,1);
end

%% Default is no step size = 1
if(nargin<6)
    eta=1;
end

%% Default is just one run
if(nargin<5)
    numiter=1;
end

%% Simulation parameters
[d, N]=size(data.training);

if r>d
    error('r should be less than d/2');
end

if q>1
    error('q should be in [0,1]')
end

%% Computing Cov matrices
CXX=(1/(size(data.testing,2)-1))*...
    (data.testing)*...
    (data.testing)';

CXXtune=(1/(size(data.tuning,2)-1))*...
    (data.tuning)*...
    (data.tuning)';


%% Check if all the runs are done
flag=1;
for method=methods
    %% IF reading from or writing to a report file
    if(sum(strcmp(method,{'batch','incremental'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,numiter=%d].mat'],...
            dataname,method{1},k,numiter);
    elseif(sum(strcmp(method,{'msg-po','oja-po','grouse'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,r=%d,numiter=%d].mat'],...
            dataname,method{1},k,eta,r,numiter);
    elseif(sum(strcmp(method,{'msg-md','oja-md'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,q=%f,numiter=%d].mat'],...
            dataname,method{1},k,eta,q,numiter);
    else
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,numiter=%d].mat'],...
            dataname,method{1},k,eta,numiter);
    end
    flag=flag && (exist(reportfile,'file'));
end

if(~flag)
    for ITER=iter_start:numiter
        
        %% Shuffle data
        rng(ITER);
        rp=randperm(N);
        data.training=data.training(:,rp);
        
        %% Sequence close to a uniform grid on semilog axis
        [seq,L]=equilogseq(N,1);

        for method=methods
            
            %% Display the run
            fprintf('Starting run: (%s,%s,%s,%d,%d)\n',...
                dataname,'PCA',method{1},k,ITER);
            
            %% Set PAGE directories
            pagepath=sprintf('../page/profile/PCA/%s/%s/',...
                method{1},dataname);
            
            if(sum(strcmp(method,{'batch','incremental'})))
                pageprefix=@(method,rank,iter)[pagepath,...
                    sprintf('%s[rank=%d,iter=%d].mat',...
                    method,rank,iter)];
            elseif(sum(strcmp(method,{'msg-po','oja-po','grouse'})))
                pageprefix=@(method,rank,iter)[pagepath,...
                    sprintf('%s[rank=%d,eta=%f,r=%d,iter=%d].mat',...
                    method,rank,eta,r,iter)];
            elseif(sum(strcmp(method,{'msg-md','oja-md'})))
                pageprefix=@(method,rank,iter)[pagepath,...
                    sprintf('%s[rank=%d,eta=%f,q=%f,iter=%d].mat',...
                    method,rank,eta,q,iter)];
            else
                pageprefix=@(method,rank,iter)[pagepath,...
                    sprintf('%s[rank=%d,eta=%f,iter=%d].mat',...
                    method,rank,eta,iter)];
            end
            % Check if the PAGE directory is structured properly
            if(~exist(pagepath,'dir'))     
                % If not create the desired directory structure
                flag=createpath(pagepath); 
                % If the directory structure could not be created
                if(~flag)                  
                    % Display error message and quit
                    error('Could not create path for result files');
                end
            end
            
            %% Output filename
            fname=pageprefix(method{1},k,ITER);
            if(~exist(fname,'file'))
%                 
%                 CXX=(1/(size(data.testing,2)-1))*...
%                     (data.testing)*...
%                     (data.testing)';
%                 
%                 CXXtune=(1/(size(data.tuning,2)-1))*...
%                     (data.tuning)*...
%                     (data.tuning)';
%                 
                %% Initialize the basis for sgd and incremental PCA
                S=0;
                if(sum(strcmp(method{1},{'sgd','oja-md','oja-po','grouse'})))
                    U=orth(randn(d,k));
                elseif(sum(strcmp(method{1},{'incremental'})))
                    U=zeros(d,k);
                    S=zeros(k,1);
                elseif(sum(strcmp(method{1},{'msg','msg-po','msg-md'})))
                    U=orth(randn(d,k));
                    S=ones(1,k)/k;
                end
                
                %% Initialize objective value
                rk=zeros(L(2),1);
                objV=zeros(L(2),1);
                objVtune=zeros(L(2),1);
                runtime = zeros(L(2),1);
                
                %% Check if we can start from a previous run
                initsamp=1;
                
                %% Loop over data
                for iter=L(1)+1:L(2)
                    fprintf('Sequence number %d...\t',seq(iter));
                    switch(method{1})
                        case 'batch'
                            %% BATCH PCA
                            isamp=seq(iter);
                            if(isamp==1)
                                continue;
                            end
                            tcounter = tic;
                            Ctrain=(1/(isamp-1))*...
                                ((data.training(:,1:isamp))*...
                                (data.training(:,1:isamp))');
                            [U,S,~]=svds(Ctrain,k);
                            U=U(:,1:k);
                            runtime(iter) = toc(tcounter);
                            rk(iter)=size(S,1);                            
                            
                        case 'sgd'
                            %% Oja's algorithm
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
                                etax=eta/(sqrt(isamp));
                                %etax = eta/(isamp);
                                tcounter = tic;
                                U=sgd(U,...
                                    data.training(:,modisamp),etax);
%                                if(~mod(modisamp,1000))
%                                    U=Gram_Schmidt(U);
%                                end
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter)=size(U,2);                                                        
                            
                        case 'incremental'
                            %% BRAND's method
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
                                tcounter = tic;
                                [U,S]=rank1update(U,S,1,data.training(:,modisamp),1e-6);
                                [U,S]=msgcapping(U,S,k);
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter)=size(U,2);
                            
                        case 'grouse'
                            %uses partial observation model
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
                                x_tilde=data.training(:,modisamp);
                                %step size for grouse~1/t
                                eta_t = 10*eta/sqrt(isamp+10);
                                rp=randperm(d);
                                x_tilde(rp(r+1:end))=0;
                                obs=rp(1:r);
                                tcounter = tic;
                                U=grouse(obs, U, eta_t, x_tilde);
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter)=size(U,2);                            
                            
                        case {'msg'}
                            %% Matrix Stochastic Gradient
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
%                                 etax=eta/sqrt(isamp);
%                                 etax=sqrt(k/N);
                                etax=eta/sqrt(isamp+10);
                                tcounter = tic;
                                [U,S]=msg(k,U,S,etax,...
                                    data.training(:,modisamp),1e-6);
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter)=length(find(S)); 
                            
                        case {'msg-po'}
                            %% Matrix Stochastic Gradient
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
%                                 etax=eta*.1/sqrt(isamp);
%                                 etax=(r*(r-1))*sqrt(k/(2*N))/sqrt(d^2*(d-1)^2+r^4*(d-r)^2);
%                                 etax=r*sqrt(k/N)/d;
                                etax=eta*r^2/(d^2*sqrt(isamp));
                                x_tilde=data.training(:,modisamp);
                                rp=randperm(d);
                                x_tilde(rp(r+1:end))=0;
                                obs=rp(1:r);
                                tcounter = tic;
                                [U,S]=msg_po(obs,k,U,S,etax,...
                                    x_tilde,1e-6);
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter)=length(find(S)); 
                            
                        case {'oja-po'}
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
                                etax = eta/sqrt(isamp);
                                x_tilde=data.training(:,modisamp);
                                rp=randperm(d);
                                x_tilde(rp(r+1:end))=0;
                                obs=rp(1:r);
                                tcounter = tic;
                                [U]=Oja_po(obs,U,etax,x_tilde);
                                runtime(iter) = toc(tcounter);
                            end
                            rk(iter) = size(U,2);
                            
                        case {'msg-md'}
                            %% Matrix Stochastic Gradient
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
%                                 etax=eta*.1/sqrt(isamp);
%                                 etax=q^2*sqrt(k/(2*N))/sqrt(q^2+d*q*(1-q)^3+d^2*q^2*(1-q)^2);
%                                 etax=q^2*2*sqrt(k/N);
                                etax=eta*q^2/sqrt(isamp);
                                x_tilde=data.training(:,modisamp);
                                rp=rand(d,1);
                                x_tilde(rp>q)=0;
                                obs=find(rp<=q);
                                if ~isempty(obs)
                                    tcounter = tic;
                                    [U,S]=msg_md(obs,q,k,U,S,etax,...
                                        x_tilde,1e-6);
                                    runtime(iter) = toc(tcounter);
                                end
                            end
                            rk(iter)=length(find(S)); 

                        case {'oja-md'}
                            for isamp=initsamp:seq(iter)
                                modisamp=1+mod(isamp-1,N);
                                etax = eta/sqrt(isamp);
                                x_tilde=data.training(:,modisamp);
                                rp=rand(d,1);
                                x_tilde(rp>q)=0;
                                obs=find(rp<=q);
                                if ~isempty(obs)
                                    tcounter =tic;
                                    [U]=Oja_md(obs,q,U,etax,x_tilde);
                                    runtime(iter) = toc(tcounter);
                                end
                            end
                            rk(iter) = size(U,2);
                    end
                    
                    initsamp=seq(iter)+1;
                    keff=min(k,size(U,2));
                    if(sum(strcmp(method{1},{'msg','msg-po','msg-md'})))
                        UU=pca_solution_original(keff,U,S);
                    else
                        UU=U;
                    end
                    if any(UU~=0)
                        UU=Gram_Schmidt(UU);
                    end
                    objVtune(iter)=trace(UU'*CXXtune*UU);
                    objV(iter)=trace(UU'*CXX*UU);
                    fprintf('\t%d\t %g\n',keff,objVtune(iter));
                end
                save(fname,'U','S','objV','objVtune','seq','rk','runtime');
            end
        end
        
    end
end

%% Ground truth
CXX=(1/(size(data.testing,2)-1))*...
    (data.testing)*...
    (data.testing)';


%% Set PAGE directories
method={'truth'};
pagepath=sprintf('../page/profile/PCA/%s/%s/',method{1},dataname);
% Check if the PAGE directory is structured properly
if(~exist(pagepath,'dir'))     
    % If not create the desired directory structure
    flag=createpath(pagepath); 
    % If the directory structure could not be created
    if(~flag)                  
        % Display error message and quit
        error('Could not create path for result files');
    end
end
fname=[pagepath,sprintf('truth[rank=%d].mat',k)];
if(~exist(fname,'file'))
%     [~,sigma]=simdist(d,1,tau);
%     objV=sum(sigma(1:k)); %#ok<NASGU>
    [EigVecs,EigVals]=eig(CXX);
    [EigVals, idx]=sort(diag(EigVals),'descend');
    objV=sum(EigVals(1:k)); %#ok<NASGU>
    EigVecs=EigVecs(:,idx(1:k));
    objVtune=trace(EigVecs'*CXXtune*EigVecs); %#ok<NASGU>
    rk=k; %#ok<NASGU>
    save(fname,'objV','objVtune','rk');
end

end
