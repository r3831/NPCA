
%% PLOTOBJV(dataname,methods,PCA,k,numpasses,ITER,AVG,PLOTREPORT) plots
% PROGRESS-PER-ITERATION (empirical & population) and PROGRESS-PER-SECOND.
%
%  The inputs are as follows:
%
%  dataname is one of these strings: 'VidTIMIT', 'VidTIMIT2', 'SIM'
%  method is a cell of strings: e.g. {'sgd', 'brand', 'batch', 'truth'}
%
%  PCA is a boolean flag - set to 1 if you want to plot PCA, 0 if CCA
%
%  k a positive integer - denotes the desired RANK
%
%  numpasses is a positive integer - denotes number of passes over the data
%
%  ITERS is an array of positve integers represening which random splits
%  use only (1-1000)
%
%  AVG is either an empty string '' in which all iterations are plotted
%  simultaneously or 'avg' in which case all iterations are avergaged
%
%  PLOTREPORT is a boolean flag - set to 0 if you want to generate a report
%  on the cluster - set to 1 if you also want to plot the report
%
%  The output containing the report is written to ../REPORT as a MAT file
%  in the following format: reportPCA[method=sgd,rank=4,numiter=10].mat
%
%  If PLOTREPORT is set to 1, three plots are generated and written to
%  ../PLOTS as pdf files with names in the following formats:
%
%  iteration[dataname=VidTIMIT,rank=1,numiter=1,...
%  methods=sgd,brand]
%  convergence[dataname=VidTIMIT,rank=1,numiter=1,...
%  methods=sgd,brand].pdf
%
%%

function plotobjV(dataname,methods,k,n,numiter,etas,d,ob_f,method_names)


if length(etas)==1
    etas=etas*ones(size(methods));
end
etas_string = sprintf('%.2f,' , etas);
etas_string = etas_string(1:end-1);

%% we assume r,q are consistent
q=1/ob_f;
r=round(d/ob_f);

%% Setup Figures
col={'k','g','b','c','m','r','y',[1 0.4 0.6]};
marker={'ks','k^','ks','kd','kh','ko','kx','k.'};
fig11=figure(11); clf; set(fig11,'Position',[2 2 1200 800]);%obj/sample
fig22=figure(22); clf; set(fig22,'Position',[2 2 1200 800]);%runtime
fig33=figure(33); clf; set(fig33,'Position',[2 2 1200 800]);%obj/observation

%% File names of figures to be plotted
fnames=cell(3,1);
fnames{1}=sprintf(['../plots/obj_iter[data=%s,rank=%d,',...
    'numiter=%d,eta=%s,r=%d,q=%f].pdf'],dataname,k,numiter,etas_string,r,q);
fnames{2}=sprintf(['../plots/time_iter[data=%s,rank=%d,',...
     'numiter=%d,eta=%s,r=%d,q=%f].pdf'],dataname,k,numiter,etas_string,r,q);
fnames{3}=sprintf(['../plots/obj_obsv[data=%s,rank=%d,',...
    'numiter=%d,eta=%s,r=%d,q=%f].pdf'],dataname,k,numiter,etas_string,r,q);
LWIDTH=6;
MSIZE=22;
maxrank=0; maxobj=0;

%% Set sequence points based on dataset and datasize
[seq,L]=equilogseq(n,1);
M=max(seq)/ob_f;
cutoff=find(seq>=M, 1 ); %plot full observation till this cutoff
maxrank=0; maxobj=0; maxtime=0; maxruntime = 0; minruntime = 10000000;

method='truth';
reportfile=sprintf(['../page/profile/PCA/%s/%s/%s[',...
    'rank=%d].mat'],method,dataname,method,k);

load(reportfile,'objV');
trueobjV=objV;


%% Plot for each method
for imethod=1:length(methods)
    
    method=methods{imethod};
    eta=etas(imethod);
    
    %% IF reading from or writing to a report file
    if(sum(strcmp(method,{'batch','incremental'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,numiter=%d].mat'],...
            dataname,method,k,numiter);
    elseif(sum(strcmp(method,{'msg-po','oja-po','grouse'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,r=%d,numiter=%d].mat'],...
            dataname,method,k,eta,r,numiter);
    elseif(sum(strcmp(method,{'msg-md','oja-md'})))
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,q=%f,numiter=%d].mat'],...
            dataname,method,k,eta,q,numiter);
    else
        reportfile=sprintf(['../page/reportPCA[data=%s,method=%s,',...
            'rank=%d,eta=%f,numiter=%d].mat'],...
            dataname,method,k,eta,numiter);
    end
    
    %% Fetch data if plotting from a report file
    if(exist(reportfile,'file'))
        load(reportfile,'seq','prog_itr','avgprogress',...
            'rank_itr','avgrank','avgtime');
        maxrank=max(max(avgrank),maxrank);
        maxobj=max(max(avgprogress),maxobj);
    else
        %% Set PAGE path and PAGE prefix
        pagepath=sprintf('../page/profile/PCA/%s/%s/',...
            method,dataname);
        
        
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
        
        %% Initialize performance metrics
        prog_itr=zeros(L(2),length(numiter));
        rank_itr=zeros(L(2),length(numiter));
        comptime = zeros(L(2),length(numiter));
        
        %% Gather performance metrics
        for iter=1:numiter
            load(pageprefix(method,k,iter),...
                'objV','objVtune','rk','runtime');
            prog_itr(:,iter)=objV(1:L(2));
            rank_itr(:,iter)=rk(1:L(2));
            comptime(:,iter)=runtime(1:L(2));
        end
        
        avgprogress=sum(prog_itr,2)./numiter;   
        avgrank=sum(rank_itr,2)./numiter;
        avgtime = sum(comptime,2)./numiter;
        save(reportfile,'seq','prog_itr','avgprogress',...
            'rank_itr','avgrank','avgtime');
        maxrank=max(max(avgrank),maxrank);
        maxobj=max(max(avgprogress),maxobj);
        maxtime=max(max(avgtime),maxtime);
        
    end
    
    avgprogress=trueobjV-avgprogress;
    runtimetotal=cumsum(avgtime);
    minruntime = min( minruntime, runtimetotal( 1 ) );
    maxruntime = max( maxruntime, runtimetotal( end ) );
    
    if(sum(strcmp(method,{'msg-po','msg-md','oja-po','oja-md','grouse'})))
        seq_obs=seq/ob_f;
        avgprogress_obs=avgprogress;
    else
        seq_obs=seq(1:cutoff);
        avgprogress_obs=avgprogress(1:cutoff);
    end
    
    fig11=figure(11);
    ignore = semilogx(seq,avgprogress,'Color',...
        col{imethod},'LineWidth',LWIDTH);
    hold on;
    set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
        'IconDisplayStyle', 'off' );
    subseq = round( (1:20) * length(seq) / 20 );
    semilogx(seq(subseq),avgprogress(subseq),marker{imethod},...
        'MarkerFaceColor',col{imethod},'MarkerSize',MSIZE);
    hold on;
    
    % %     fig22=figure(22);
    % %     ignore = semilogx(seq,avgrank,'Color',...
    % %         col{imethod},'LineWidth',LWIDTH);
    % %     hold on;
    % %     set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
    % %         'IconDisplayStyle', 'off' );
    % %     subseq = round( (1:20) * length(seq) / 20 );
    % %     semilogx(seq(subseq),avgrank(subseq),marker{imethod},...
    % %         'MarkerFaceColor',col{imethod},'MarkerSize',MSIZE);
    % %     hold on;
    
    fig22 = figure(22);
    ignore = semilogx(runtimetotal,avgprogress,'Color',...
        col{imethod},'LineWidth',LWIDTH);
    hold on;
    set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
        'IconDisplayStyle', 'off' );
    subseq = round( (1:20) * length(runtimetotal) / 20 );
    semilogx(runtimetotal(subseq),avgprogress(subseq),marker{imethod},...
        'MarkerFaceColor',col{imethod},'MarkerSize',MSIZE);
    hold on;
    
    fig33=figure(33);
    ignore = semilogx(seq_obs,avgprogress_obs,'Color',...
        col{imethod},'LineWidth',LWIDTH);
    hold on;
    set( get( get( ignore, 'Annotation' ), 'LegendInformation' ),...
        'IconDisplayStyle', 'off' );
    subseq = round( (1:20) * length(seq_obs) / 20 );
    semilogx(seq_obs(subseq),avgprogress_obs(subseq),marker{imethod},...
        'MarkerFaceColor',col{imethod},'MarkerSize',MSIZE);
    hold on;
    
%     fig44=figure(4);
%     semilogx(seq_obs,avgprogress_obs,marker{imethod},...
%         'MarkerFaceColor',col{imethod},'MarkerSize',MSIZE);
%     hold on;
    
end

%% PLOT TRUTH (if reading from a report then nothing to do)
method='truth';
reportfile=sprintf(['../page/profile/PCA/%s/%s/%s[',...
    'rank=%d].mat'],method,dataname,method,k);

load(reportfile,'objV');
trueobjV=objV;

FSIZE1=40; %70; %20; 
FSIZE2=40; %70; %30;
FSIZE3=40; %40; %30;

figure(11); %semilogx(seq,repmat(trueobjV,size(seq)),'k-','LineWidth',LWIDTH); hold on;
grid; xlabel('Iteration','FontSize',FSIZE2);%,'Interpreter','Latex');
maxobj=max(maxobj,trueobjV);
axis([0 seq(end) 0 inf]);
ylabel('Objective','FontSize',FSIZE2);%,'Interpreter','Latex');
set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
    'XMinorGrid','off','YScale','log');

% % figure(22); hold on; grid; xlabel('Iteration','FontSize',FSIZE2);
% % axis([0 seq(end) 0 maxrank+1]);
% % ylabel('Rank of iterates','FontSize',FSIZE2);%,'Interpreter','Latex');
% % set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
% %     'XMinorGrid','off');

figure(22);
hold on; grid; 
xlabel('Time','FontSize',FSIZE2);
ylabel('Objective','FontSize',FSIZE2);%,'Interpreter','Latex');
% axis([0 seq_obs(end) 0 1]);
axis([0 inf 0 inf]);
set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
    'XMinorGrid','off', 'YScale', 'log');


figure(33); %semilogx(seq_obs,repmat(trueobjV,size(seq_obs)),'k-','LineWidth',LWIDTH); hold on;
grid; xlabel('Observations','FontSize',FSIZE2);%,'Interpreter','Latex');
maxobj=max(maxobj,trueobjV);
axis([0 seq_obs(end) 0 inf]);
ylabel('Objective','FontSize',FSIZE2);%,'Interpreter','Latex');
set(gca,'FontSize',FSIZE1,'XTick',[10^0 10^1 10^2 10^3 10^4 10^5],...
    'XMinorGrid','off','YScale','log');
if(q>.11)
    mylegend=legend(method_names,'Location','Southwest');
    set(mylegend,'FontSize',FSIZE3);
end
%if(k==2&&ob_f==2)
%     mylegend=legend([methods,'Truth'],'Location','Southwest');
%     set(mylegend,'FontSize',FSIZE3);
%end

%% Create PDFs 
topdf(fig11,fnames{1});
topdf(fig22,fnames{2});
topdf(fig33,fnames{3});
end
