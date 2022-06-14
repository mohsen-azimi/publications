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
myCase = 3;
name = ['A_SHM',num2str(myCase),'_TH']; % Bridge/SHM
% name = ['A_SHM',num2str(myCase),'_freq']; xLim = 100;% Bridge/SHM
nClasses = numel(myCasesTable{myCase,3}); 


%%



load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',name,'.mat']);


xLim = 1000;
figure(1); set(figure(1), 'position',[150         200        800         500])
if exist('acc')==1
    AccuracyTH = acc;
    LossTH     = loss;
end
yyaxis left;  
        plot(AccuracyTH(1:min(xLim,numel(AccuracyTH)))*100,'b','LineWidth',2); grid off
        xlabel('Epoch','FontWeight','bold','fontsize',15,'fontname','Times New Roman')
        ylabel('Accuracy (%)','FontWeight','bold','fontsize',15,'fontname','Times New Roman','Interpreter', 'tex')

pyyL = gca;
      pyyL.YColor = [0 0 1];
  
        
yyaxis right;  
        plot(LossTH(1:min(xLim,numel(LossTH))),'r','LineWidth',2); grid off
        xlabel('Epoch','FontWeight','bold','fontsize',16,'fontname','Times New Roman')
        ylabel('Loss','FontWeight','bold','fontsize',16,'fontname','Times New Roman','Interpreter', 'tex')
        
%         title([name,'_{',num,'}'],'FontWeight','bold','fontsize',18,'fontname','Times New Roman')
 
pyyR = gca;
      pyyR.YColor = [1 0 0];
      pyyR.FontName = 'Times New Roman';
      pyyR.FontSize = 18;
        
%% Plot Confusion Matrix


trueVector = Y_true+1;% True classes  %+1 for indexing match with Matlab
predVector = Y_pred+1;% Predicted classes





% Convert this data to a [nClasses x 6] matrix
y_true = zeros(nClasses,numel(trueVector));
y_pred = zeros(nClasses,numel(predVector));

targetsIdx = sub2ind(size(y_true), trueVector, 1:numel(trueVector));
outputsIdx = sub2ind(size(y_pred), predVector, 1:numel(predVector));

y_true(targetsIdx) = 1;
y_pred(outputsIdx) = 1;
% Plot the confusion matrix for a 3-class problem

figure(123)
% f_plotconfusion(y_true,y_true,'')
f_plotconfusion(y_true,y_pred,'')


% plotconfusion(y_true,y_pred,'Case 2: ')

.........................................................................
h = gca;
    set(h, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',12) 
    h.XTickLabel = {'Intact','P1','P2','P3','P4','P5','P6','P7','P8',' '};
%     h.XTickLabel = {'Intact','Pattern 1','Pattern 2','Pattern 3','Pattern 4','Pattern 5','Pattern 6','Pattern 7','Pattern 8',' '};
    h.YTickLabel = h.XTickLabel; %{'Intact','P1','P2','P3','P4','P5','P6','P7','P8',' '};

    
    
         if strcmp(name(end-3:end),'freq')==1
            appendName = ', Self-powered Sensors';
        else
            appendName = ', MEMS Sensors';
        end

%         title(['Confusion Matrix for Case ',num2str(myCase),appendName],'FontWeight','bold', 'fontsize',12,'fontname','Times New Roman','Interpreter', 'none')
   
    
        
        xlabel(h,'True Class','FontWeight','bold','fontsize',14,'fontname','Times New Roman')
        ylabel(h,'Predicted Class','FontWeight','bold','fontsize',14,'fontname','Times New Roman','Interpreter', 'tex')
        title(['Case ',num2str(myCase),appendName],'FontWeight','bold', 'fontsize',14,'fontname','Times New Roman','Interpreter', 'none')


%% Copy Confusion matrix to Figure(1) subplot

aH(1) = gca;
acH(1) = {flipud(allchild(aH(1)))};
...........................................................
figure(1)
 p = get(gca, 'Position');
 sH = axes('Position', [p(1)+.20 p(2)+.10 p(3)-.25 p(4)-.20]); % 3 classes

...........................................................
for ii = 1:numel(acH{1})
    copyobj(acH{1}(ii),sH(1));
end

set(sH,'Visible','on');
set(sH,'Ydir','reverse')
 

        if strcmp(name(end-3:end),'freq')==1
            appendName = ', Self-powered Sensors';
        else
            appendName = ', MEMS Sensors';
        end

        


sH.XLabel.Visible = 'on'; 
sH.YLabel.Visible = 'on'; 
sH.Title.Visible  = 'on';

switch nClasses
    case 2
%     sH.XTickLabel = {' ',' ',' ','Intact',' ',['P',num2str(myCasesTable{myCase,3}(2))],' ',' ',' ',' '};
    case 3
    sH.XTickLabel = {' ','Intact',' ',['P',num2str(myCasesTable{myCase,3}(2))],' ',['P',num2str(myCasesTable{myCase,3}(3))],' ',' ',' ',' '};
    case 5
    sH.XTickLabel = {'Intact',['P',num2str(myCasesTable{myCase,3}(2))],['P',num2str(myCasesTable{myCase,3}(3))],['P',num2str(myCasesTable{myCase,3}(4))],...
        ['P',num2str(myCasesTable{myCase,3}(5))],' ',' ',' ',' '};
    
    axis([.5 6.5 .5 6.5]) % to fix the matrix

    case 9
    sH.XTickLabel = {' ','Intact',['P',num2str(myCasesTable{myCase,3}(2))],['P',num2str(myCasesTable{myCase,3}(3))],['P',num2str(myCasesTable{myCase,3}(4))],...
        ['P',num2str(myCasesTable{myCase,3}(5))],['P',num2str(myCasesTable{myCase,3}(6))],['P',num2str(myCasesTable{myCase,3}(7))],['P',num2str(myCasesTable{myCase,3}(8))],['P',num2str(myCasesTable{myCase,3}(9))],' ',' ',' ',' '};
end
    sH.YTickLabel = sH.XTickLabel;

%     sH.FontSize = 11;
    sH.FontWeight='bold';
    
        title(['Case ',num2str(myCase),appendName],'FontWeight','bold', 'fontsize',14,'fontname','Times New Roman','Interpreter', 'none')
        xlabel(sH,'True Class','FontWeight','bold','fontsize',14,'fontname','Times New Roman')
        ylabel(sH,'Predicted Class','FontWeight','bold','fontsize',14,'fontname','Times New Roman','Interpreter', 'tex')
   

sH.XRuler.Axle.LineStyle = 'none';  
sH.YRuler.Axle.LineStyle = 'none';

sH.XTickLabelRotation = 0;
sH.YTickLabelRotation = 90;

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
% % % 
% % % % % 





%%
% % % load fisheriris
% % % X_train = meas(51:end,1:2);
% % % y_train = (1:100)'>50;  % versicolor = 0, virginica = 1
% % % 
% % % cls     = species(51:end,:);
% % % posClass = 'virginica';
% % % mdl = fitglm(X_train,y_train,'Distribution','binomial','Link','logit');    
% % % 
% % % scores = mdl.Fitted.Probability;
% % % [X,Y,T,AUC] = perfcurve(cls,scores,posClass); 
% % % 
% % % figure(55)
% % % plot(X,Y)
% % % xlabel('False positive rate') 
% % % ylabel('True positive rate')

