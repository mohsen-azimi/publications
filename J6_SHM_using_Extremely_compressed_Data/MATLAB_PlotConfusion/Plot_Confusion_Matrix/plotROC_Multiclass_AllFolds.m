clear; close all; clc
%%
myCasesTable = {[1],     [1],   [0, 1, 2];
    [2],     [1],   [0, 3, 4];
    [3],     [2],   [0, 1, 2, 3, 4];
    
    [4],     [1],   [0, 5, 6];
    [5],     [2],   [0, 5, 6];
    
    [6],     [3],   [0, 5, 6];
    
    [7],     [2],   [0, 7, 8];
    [8],     [2],   [0, 1, 2, 3, 4, 5, 6, 7, 8]};
%%
myCase = 8

for fig = 1:3  % three methods of damage detection
    set(figure(fig), 'Position', [300   100   800   500])
    
    
    switch fig
        case 1
            name = ['A_SHM',num2str(myCase),'_TH']; % TH/freq/2
            figLegend = ['Case ',num2str(myCase),', MEMS'];
        case 2
            name = ['A_SHM',num2str(myCase),'_freq']; % TH/freq/2
            figLegend = ['Case ',num2str(myCase),', Self-powered'];
        case 3
            name = ['TL_A_SHM',num2str(myCase),'_freq2']; % TH/freq/2
            figLegend = ['Case ',num2str(myCase),', Transfer Learning'];
    end
    
    
    nClasses = numel(myCasesTable{myCase,3});
    
    
    %% collect all folds
    fold = 1;
        load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',name,'_fold',num2str(fold),'.mat']);
        Yps = Y_predScores;
        Yt  = Y_true;
        Yp  = Y_pred;
     
     for fold = 2:10
         load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',name,'_fold',num2str(fold),'.mat']);
         Yps = [Yps; Y_predScores];
         Yt  = [Yt, Y_true];
         Yp  = [Yp, Y_pred];
     end
    
    Y_predScores = Yps; clear Yps;
    Y_true       = Yt;  clear Yt;
    Y_pred       = Yp;  clear Yp;
    
    
    %%
    
    
    
    figure(fig); hold on, box on
    
    for cls = 0:(nClasses-1)
        index = find([0:(nClasses-1)]~=cls); Max = zeros(size(Y_predScores,1),1);
        for i = 1:numel(index)
            Max = max(Max, Y_predScores(:,index(i)));
        end
        diffscore = Y_predScores(:,cls+1) - Max;
        
        [X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(Y_true',diffscore,double(cls));
        
        if cls == 0
            plot(X,Y,':k','linewidth',2,'displayname',['Intact,        AUC = ',num2str(AUC,'%3.3f')])
        else
            plot(X,Y,'linewidth',2,'displayname',['Pattern ',num2str(cls),',   AUC = ',num2str(AUC,'%3.3f')])
        end
        
        axis([-.01 1.01 -.01 1.01])
        
        
    end
    
    set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',12)
    
    xlabel('False Positive Rate', 'fontsize',16, 'fontname','Times New Roman','FontWeight','Bold')
    ylabel('True Positive Rate', 'fontsize',16, 'fontname','Times New Roman','FontWeight','Bold')
    title('Receiver Operating Characteristic (ROC) Curves', 'fontsize',18, 'fontname','Times New Roman','FontWeight','Bold')
    
    
    text(.6,.1,figLegend, 'fontsize',14, 'fontname','Times New Roman','FontWeight','Bold')
    
    
    %               ylim([-.1 1.1]);
    %                title('Sigmoid:    $f(x)=1/({1+e^{-x}})$',...
    %                     'HorizontalAlignment','center','fontsize',10,...
    %                     'fontname','Times New Roman','FontWeight','Bold',...
    %                     'EdgeColor','none','BackgroundColor','none','LineWidth',1,'interpreter','latex')
    
    
    
    
    
    legend('show', 'location','southwest','orientation','vertical','fontsize',14)
    
    
    
    %% Copy and Zoom
    
    
    aH(1) = gca;
    acH(1) = {flipud(allchild(aH(1)))};
    ...........................................................
        figure(fig)
    p = get(gca, 'Position');
    %sH = axes('Position', [p(1)+.3 p(2)+.2 p(3)-.4 p(4)-.4]); % 3 classes
    sH = axes('Position', [p(1)+.36 p(2)+.37    p(3)-.375 p(4)-.39]); % 6 classes
    
    ...........................................................
        for ii = 1:numel(acH{1})
        copyobj(acH{1}(ii),sH(1));
        end
        set(sH,'Visible','on');
        % set(sH,'Ydir','reverse')
        axis([-.01 .2 .8 1.01])
        axis on; box on
        set(gca, 'LineWidth',1, 'FontWeight','normal',...
            'FontName','Times New Roman', 'FontSize',12)
        
        ax = gca; ax.XColor = 'b'; ax.YColor = 'b';
        
        %% Annotate
        annotation('arrow',[.3 .44],[.8345 .7],'Color','b','linewidth',1);
        annotation('rectangle',[.133 .7648 .15 .156],'Color','b','linewidth',1)
        
        
end % fig


%% ROC Table
tableROC = {};

for myCase = 2:8
    for method = 1:3  % three methods of damage detection
        switch method
            case 1
                name = ['A_SHM',num2str(myCase),'_TH']; % TH/freq/2
                figLegend = ['Case ',num2str(myCase),', MEMS'];
            case 2
                name = ['A_SHM',num2str(myCase),'_freq']; % TH/freq/2
                figLegend = ['Case ',num2str(myCase),', Self-powered'];
            case 3
                name = ['A_SHM',num2str(myCase),'_freq2']; % TH/freq/2
                figLegend = ['Case ',num2str(myCase),', Transfer Learning'];
        end
        nClasses = numel(myCasesTable{myCase,3});
        %%
        load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',name,'.mat']);
        %%
        
        for cls = 0:(nClasses-1)
            index = find([0:(nClasses-1)]~=cls); Max = zeros(size(Y_predScores,1),1);
            for i = 1:numel(index)
                Max = max(Max, Y_predScores(:,index(i)));
            end
            diffscore = Y_predScores(:,cls+1) - Max;
            [X,Y,T,AUC,OPTROCPT,suby,subnames] = perfcurve(Y_true',diffscore,double(cls));
            tableROC{myCase,1}(cls+1,method) = AUC;
        end
    end % fig
end % case




