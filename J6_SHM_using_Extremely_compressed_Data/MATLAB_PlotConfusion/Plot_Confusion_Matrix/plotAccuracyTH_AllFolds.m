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
myCase = [7]; % Change fonts ;-)
name = ['A_SHM',num2str(myCase),'_TH']; % Bridge/SHM
% name = ['A_SHM',num2str(myCase),'_freq']; % Bridge/SHM
% name = ['TL_A_SHM',num2str(myCase),'_freq1']; % Bridge/SHM

nClasses = numel(myCasesTable{myCase,3}); 


%% load 1st fold 
load(['G:\My Drive\Research\Publications\03 Published\2019_CACAIE\AISC_BenchMark\Python\saveMATs\',name,'_fold',num2str(1),'.mat']);

%% Read all data 

for fold = 2:nFolds
    

    Loaded = load(['G:\My Drive\Research\Publications\03 Published\2019_CACAIE\AISC_BenchMark\Python\saveMATs\',name,'_fold',num2str(fold),'.mat']);


    AccuracyTH = [AccuracyTH; Loaded.AccuracyTH];   % stack on top of each other
    
    LossTH     = [LossTH; Loaded.LossTH];           % stack on top of each other

    cnf_matrix = cnf_matrix + Loaded.cnf_matrix;
    
    Y_pred = [Y_pred , Loaded.Y_pred];
    Y_true = [Y_true , Loaded.Y_true];
end

% % Save for metrics calculations
% save(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMetrics',name]);

    % mean and max
    minAccTH   = min(AccuracyTH,[],1);
    meanAccTH  = mean(AccuracyTH,1);
    maxAccTH   = max(AccuracyTH,[],1);
    minLossTH  = min(LossTH,[],1);
    meanLossTH = mean(LossTH,1);
    maxLossTH  = max(LossTH,[],1);

%% 
xLim = 1000;
figure(1); set(figure(1), 'position',[150         200        800         500])
% if exist('acc')==1; AccuracyTH = acc; LossTH = loss; end

yyaxis left;  
        
        x = 1:nepochs;
        fill([x flip(x)],[minAccTH flip(maxAccTH)]*100,'b','LineStyle','none','FaceColor','b','FaceAlpha',.3)
        grid off; hold on
%         plot(AccuracyTH'*100,':b','LineWidth',1); 
        plot(meanAccTH*100,'-b','LineWidth',2); 

        xlabel('Epoch','FontWeight','bold','fontsize',15,'fontname','Times New Roman')
        ylabel('Accuracy (%)','FontWeight','bold','fontsize',15,'fontname','Times New Roman','Interpreter', 'tex')
% xlim([0 500])
pyyL = gca;
      pyyL.YColor = [0 0 1];
  
        
yyaxis right;  
        
        x = 1:nepochs;
        fill([x flip(x)],[minLossTH flip(maxLossTH)],'r','LineStyle','none','FaceColor','r','FaceAlpha',.3)
        grid off; hold on
        plot(meanLossTH,'-r','LineWidth',2);

        xlabel('Epoch','FontWeight','bold','fontsize',16,'fontname','Times New Roman')
        ylabel('Loss','FontWeight','bold','fontsize',16,'fontname','Times New Roman','Interpreter', 'tex')
        
%         title([name,'_{',num,'}'],'FontWeight','bold','fontsize',18,'fontname','Times New Roman')
%  xlim([0 500])

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





