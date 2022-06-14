clear; close all; clc
%%
myCasesTable = {[1],     [2],   [0, 1];
                [2],     [2],   [0, 2];
                [3],     [2],   [0, 3];
                [4],     [2],   [0, 4];
                [5],     [2],   [0, 5];
                [6],     [2],   [0, 6];
                [7],     [2],   [0, 7];
                [8],     [2],   [0, 8]};
           
%%

for myCase = 1:1
name = ['Binary_SHM',num2str(myCase)]; 
nClasses = numel(myCasesTable{myCase,3});

load(['C:\MOHSEN\Research\Temp\03_CNN\Python\CNN\saveMATs\',name,'.mat']);


%% ROC
% % % figure()
% % % f_plotroc(y_true,y_pred,['title'])
% % % 
% % % leg = legend;
% % % set(leg, 'location','southeast')
% % % 
% % % box on; grid on;
% % % set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)               
% % %     xlabel('False Positive Rate', 'fontsize',16, 'fontname','Times New Roman','FontWeight','Bold')
% % %     ylabel('True Positive Rate', 'fontsize',16, 'fontname','Times New Roman','FontWeight','Bold')
% % %     title(['ROC'], 'fontsize',16, 'fontname','Times New Roman','FontWeight','Bold')
% % % 
% % %     
% % %      text (.7,.4,'SNR = [   ]','fontsize',16, 'fontname','Times New Roman','FontWeight','Bold', 'HorizontalAlignment','Center')
%%    
... [X,Y,T,AUC] = perfcurve(labels,scores,posclass)
    [X,Y,T,AUC] = perfcurve(Y_true',Y_predScores(:,2),1); 

figure(2)
plot(X,Y); hold on

xlabel('False positive rate') 
ylabel('True positive rate')
legend('show')

end

