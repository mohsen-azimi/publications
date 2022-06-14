clear; close all; clc
%% https://scikit-learn.org/stable/modules/model_evaluation.html
myCasesTable = {[1],     [1],   [0, 1, 2];
                [2],     [1],   [0, 3, 4];
                [3],     [2],   [0, 1, 2, 3, 4];
                
                [4],     [1],   [0, 5, 6];
                [5],     [2],   [0, 5, 6];
                
                [6],     [3],   [0, 5, 6];
                
                [7],     [2],   [0, 7, 8];
                [8],     [2],   [0, 1, 2, 3, 4, 5, 6, 7, 8]};

%% Make subplots tight

make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

for names = 1:3



for myCase = [1:8] % Change fonts ;-)
% nClasses = numel(myCasesTable{myCase,3}); 
    switch names
        case 1
            name = ['A_SHM',num2str(myCase),'_TH'];  ttl = ['Case ',num2str(myCase)];%,', MEMS Sensors'];
        case 2
            name = ['A_SHM',num2str(myCase),'_freq']; ttl = ['Case ',num2str(myCase)];%,', Self-powered Sensors'];
        case 3
            name = ['TL_A_SHM',num2str(myCase),'_freq2'];  ttl = ['Case ',num2str(myCase)];%,', Compressed TL'];
    end

%% Read all data 
for fold = 1:10
    
    load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMetrics\',name,'_fold',num2str(fold),'.mat']);
    
    table(fold,1) = fold;
    
    table(fold,2) = Accuracy_score;
    table(fold,3) = f1_score;
    table(fold,4) = precision_score;
    table(fold,5) = recall_score;
    
end
figure(names);  set(figure(names), 'position',[20         100        1300         400])
            subplot(2,4,myCase);

boxplot(table(:,2:end)*100,'Colors','k','symbol','k+');
        box on; grid off; 

        if myCase==1
            set(gca,'XTickLabel', {' ';' ';' ';' '});
        elseif myCase==2
            set(gca,'XTickLabel', {' ';' ';' ';' '});
        elseif myCase==3
            set(gca,'XTickLabel', {' ';' ';' ';' '});
        elseif myCase==4
            set(gca,'XTickLabel', {' ';' ';' ';' '});
        else
            set(gca,'XTickLabel', {'Acc.';'Precision';'Recall';'F1'});
        end
            
            
        
        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
%         xlabel('Metric','FontWeight','bold','fontsize',10,'fontname','Times New Roman')
        
        if myCase==1
        ylabel('Score (%)','FontWeight','bold','fontsize',10,'fontname','Times New Roman','Interpreter', 'tex')
        elseif myCase==5
        ylabel('Score (%)','FontWeight','bold','fontsize',10,'fontname','Times New Roman','Interpreter', 'tex')
        end

%        ylim([95 100.6]) % TH
..........................................
%         ylim([90 100.6]) % freq
%         if myCase==4
%         ylim([85 100.6])
%         end
..........................................
        ylim([95 100.6])  % TL
        if myCase==3
        ylim([50 100.6])
        end
..........................................
        Y = ylim;
        title(ttl,'Position', [2.5, (Y(1)+.05*(Y(2)-Y(1)))],'FontWeight','bold','fontsize',10,'fontname','Times New Roman','Interpreter', 'none')

        tix=get(gca,'ytick')';
        set(gca,'yticklabel',num2str(tix,'%0.0f'))



end
end





 %% make subplots closer
 

 
 
