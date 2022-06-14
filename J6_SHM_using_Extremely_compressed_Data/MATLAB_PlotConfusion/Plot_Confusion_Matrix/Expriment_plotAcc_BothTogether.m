clear; close all; clc


for iFile = 1:2

%% Read all data 
    if iFile ==1
        file = 'Expr_allTHData_shm_a_freq_fold'; nFolds=5; folds = [2:nFolds];
        Title = ['IASC-ASCE'];
    else
        file = 'Expr_Avci_B_freq_fold'; nFolds = 5; folds = [1, 1, 4, 5];
        Title = ['QUGS'];

    end
fold = 1;
    load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',file,num2str(1),'.mat']);
for fold = folds
  
    Loaded = load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',file,num2str(fold),'.mat']);
  



    AccuracyTH = [AccuracyTH; Loaded.AccuracyTH];   % stack on top of each other
    AccuracyTH_val = [AccuracyTH_val; Loaded.AccuracyTH_val];   % stack on top of each other
    
    LossTH     = [LossTH; Loaded.LossTH];           % stack on top of each other
    LossTH_val     = [LossTH_val; Loaded.LossTH_val];           % stack on top of each other

    cnf_matrix = cnf_matrix + Loaded.cnf_matrix;
    
    scores = [scores; Loaded.scores];
    
end

    % mean and max
    minAccTH   = min(AccuracyTH,[],1);
    meanAccTH  = mean(AccuracyTH,1);
    maxAccTH   = max(AccuracyTH,[],1);
    minLossTH  = min(LossTH,[],1);
    meanLossTH = mean(LossTH,1);
    maxLossTH  = max(LossTH,[],1);

    minAccTH_val   = min(AccuracyTH_val,[],1);
    meanAccTH_val  = mean(AccuracyTH_val,1);
    maxAccTH_val   = max(AccuracyTH_val,[],1);
    minLossTH_val  = min(LossTH_val,[],1);
    meanLossTH_val = mean(LossTH_val,1);
    maxLossTH_val  = max(LossTH_val,[],1);

%% Plot TH
figure(1); set(figure(1), 'Position', [100   100   400   500])

subplot(2,1,iFile)
grid on; hold on; box on
        

if iFile==1
        % SHM
        plot(meanAccTH*100,'-k','LineWidth',2,'color',[0 00 1])
        plot(meanAccTH_val*100,'-k','LineWidth',2, 'color', [1 0.0 0.0])
else
        % Avci
        plot(smooth(meanAccTH*100,    10,'rloess'),'-k','LineWidth',2,'color',[0 0 1])
        plot(smooth(maxAccTH_val*100, 10,'rloess'),'-k','LineWidth',2, 'color', [1 0 0])
        legend('Training','Validation', 'location','southeast');

end
        set(gca, 'FontWeight','normal','fontsize',12,'fontname','Times New Roman','linewidth',1)
        xlabel('Epoch','FontWeight','bold','fontsize',12,'fontname','Times New Roman')
        ylabel('Accuracy (%)','FontWeight','bold','fontsize',12,'fontname','Times New Roman','Interpreter', 'tex')
        text (250,50,Title,'HorizontalAlignment','center','fontsize',10,'fontname','Times New Roman');
       
ylim([0 101])



 




end
%% Plot Confusion Matrix

% figure(2); set(figure(2), 'position',[100         100        1000         600])
% % figure(2); set(figure(2), 'position',.5*[500         500        1000         600])
% subplot(1,2,2)
% 
% h = heatmap((cnf_matrix));
% 
% h.Title = ['Confusion Matrix (Accuracy = ',num2str(mean(scores(:,2))*100,'%0.1f'),' %)'];
% h.XLabel = 'True Class';
% h.YLabel = 'Predicted Class';
% h.FontName = 'Times New Roman';
% h.FontSize = 12;
% h.ActivePositionProperty = 'Position';
% h.Position = [0.05 0.1 0.85 0.85];
% % h.Position = [0.1 0.15 0.75 0.75];
% 
% %% Plot side-by-side
