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
figure(iFile)
h = heatmap((cnf_matrix));

if iFile ==1
    % ASCE
    figure(iFile); set(figure(iFile), 'position',.5*[100         100        1000         600])
    h.Position = [0.1 0.15 0.85 0.75];

else
    % Avci
    figure(iFile); set(figure(iFile), 'position',1*[100         100        1000         600])
    h.Position = [0.05 0.1 0.92 0.85];


    
end



h.Title = [Title, ' (Acc. = ',num2str(mean(scores(:,2))*100,'%0.1f'),' %)'];
h.XLabel = 'True Class';
h.YLabel = 'Predicted Class';
h.FontName = 'Times New Roman';
h.FontSize = 12;
h.ActivePositionProperty = 'Position';
h.ColorbarVisible = 'off';
h.ColorScaling = 'scaledcolumns';

% g = h.Colormap; g(:,1)= linspace(1,0,size(g,1));g(:,2)=g(:,1);g(:,3)=g(:,1);
% h.Colormap = g;
end
%% Plot Confusion Matrix
% % % 
% % % figure(2); set(figure(2), 'position',[100         100        1000         600])
% % % % figure(2); set(figure(2), 'position',.5*[500         500        1000         600])
% % % subplot(1,2,2)
% % % 
% % % h = heatmap((cnf_matrix));
% % % 
% % % h.Title = ['Confusion Matrix (Accuracy = ',num2str(mean(scores(:,2))*100,'%0.1f'),' %)'];
% % % h.XLabel = 'True Class';
% % % h.YLabel = 'Predicted Class';
% % % h.FontName = 'Times New Roman';
% % % h.FontSize = 12;
% % % h.ActivePositionProperty = 'Position';
% % % h.Position = [0.05 0.1 0.85 0.85];
% % % % h.Position = [0.1 0.15 0.75 0.75];

% % % %% Plot side-by-side
% % % %% Plot TH
% % % xLim = 100;
% % % figure(3); set(figure(3), 'position',[10         10        1300     600])
% % % % figure(3); set(figure(3), 'position',.5*[500         500        2*1000         600])
% % % 
% % %  subplot(1,2,1)
% % % grid on; hold on; box on
% % %         
% % %         plot(smooth(meanAccTH*100,1,'rloess'),'-b','color',[0 0.4470 0.7410],'LineWidth',2); 
% % %         plot(smooth(meanAccTH_val*100,1,'rloess'),'-b','color', [0.8500 0.3250 0.0980],'LineWidth',2); 
% % % 
% % %          set(gca, 'FontWeight','normal','fontsize',12,'fontname','Times New Roman')
% % %         xlabel('Epoch','FontWeight','bold','fontsize',12,'fontname','Times New Roman')
% % %         ylabel('Accuracy (%)','FontWeight','bold','fontsize',12,'fontname','Times New Roman','Interpreter', 'tex')
% % %        
% % % ylim([0 101])
% % % 
% % % g = gca; 
% % % g.ActivePositionProperty = 'Position';
% % % g.Position = [0.05 0.15 0.4 0.78];
% % % 
% % % legend('Training','Validation', 'location','southeast');
% % %  
% % % % Plot Confusion Matrix
% % % 
% % % subplot(1,2,2)
% % % h = heatmap(cnf_matrix);
% % % 
% % % 
% % % h.Title = ['Confusion Matrix (Accuracy = ',num2str(mean(scores(:,2))*100),' %)'];
% % % h.XLabel = 'True Class';
% % % h.YLabel = 'Predicted Class';
% % %    
% % % h.FontName = 'Times New Roman';
% % % h.FontSize = 12;
% % % 
% % % h.ActivePositionProperty = 'Position';
% % % h.Position = [0.52 0.15 0.4 0.78];
% % % 
