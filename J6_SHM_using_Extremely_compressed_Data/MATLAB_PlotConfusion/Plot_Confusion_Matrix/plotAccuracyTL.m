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

for myCase = 4
files = {['A_SHM',num2str(myCase),'_freq1'];
         ['A_SHM',num2str(myCase),'_freq2_val']}; % 
     
nClasses = numel(myCasesTable{myCase,3});

blueLine = {'-b';'--b'};
redLine  = {'-r';'--r'};

accLegends  = {'Source Domain, Training Accuracy';
               'Target Domain,  Training Accuracy '};
lossLegends = {'Source Domain, Training Loss';
               'Target Domain,  Training Loss'};

%%

for file = 1:2
     
load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',files{file},'.mat']);

figure(myCase); set(figure(myCase), 'position',[150         200        800         500])


yyaxis left;  
        plot(AccuracyTH(1:min(1000,numel(AccuracyTH)))*100,blueLine{file},'displayname',accLegends{file},'LineWidth',2); grid off; hold on
        xlabel('Epoch','FontWeight','bold','fontsize',16,'fontname','Times New Roman')
        ylabel('Accuracy (%)','FontWeight','bold','fontsize',16,'fontname','Times New Roman','Interpreter', 'tex')

pyyL = gca;
      pyyL.YColor = [0 0 1];
      
      ylim([-0 100])
        
yyaxis right;  
        plot(LossTH(1:min(1000,numel(LossTH))),redLine{file},'displayname',lossLegends{file},'LineWidth',2); grid off; hold on
        xlabel('Epoch','FontWeight','bold','fontsize',18,'fontname','Times New Roman')
        ylabel('Loss','FontWeight','bold','fontsize',18,'fontname','Times New Roman','Interpreter', 'tex')
        
%         title([name,'_{',num,'}'],'FontWeight','bold','fontsize',18,'fontname','Times New Roman')
 
pyyR = gca;
      pyyR.YColor = [1 0 0];
      pyyR.FontName = 'Times New Roman';
      pyyR.FontSize = 18;
        
      
end

legend('show', 'location','east','fontsize',18,...
        'fontname','Times New Roman');
% 
%    
   
   
   
   Title = {['{\bfCase ',num2str(myCase),'}']};
         
         
    X=xlim; Y=ylim;
    text(.65*X(2),.69*Y(2),Title,...
        'HorizontalAlignment','center','fontsize',18,...
        'fontname','Times New Roman','FontWeight','Bold',...
        'EdgeColor','w','BackgroundColor','w','LineWidth',1)

    
    
    grid on

    
    
end  
%% Plot Confusion Matrix

% % % 
% % % trueVector = Y_true+1;% True classes  %+1 for indexing match with Matlab
% % % predVector = Y_pred+1;% Predicted classes
% % % 
% % % 
% % % 
% % % 
% % % 
% % % % Convert this data to a [nClasses x 6] matrix
% % % y_true = zeros(nClasses,numel(trueVector));
% % % y_pred = zeros(nClasses,numel(predVector));
% % % 
% % % targetsIdx = sub2ind(size(y_true), trueVector, 1:numel(trueVector));
% % % outputsIdx = sub2ind(size(y_pred), predVector, 1:numel(predVector));
% % % 
% % % y_true(targetsIdx) = 1;
% % % y_pred(outputsIdx) = 1;
% % % % Plot the confusion matrix for a 3-class problem
% % % 
% % % figure(123)
% % % % f_plotconfusion(y_true,y_pred,'')
% % % plotconfusion(y_true,y_pred,'Case 2: ')
% % % 
% % % .........................................................................
% % % h = gca;
% % %     set(h, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
% % %     h.XTickLabel = {'Intact','P1','P2','P3','P4','P5','P6','P7','P8',' '};
% % %     h.YTickLabel = {'Intact','P1','P2','P3','P4','P5','P6','P7','P8',' '};

    
    
% % %          if strcmp(name(end-3:end),'freq')==1
% % %             appendName = ', Self-powered Sensors';
% % %         else
% % %             appendName = ', MEMS Sensors';
% % %         end
% % % 
% % %         title(['Confusion Matrix for Case ',num2str(myCase),appendName],'FontWeight','bold', 'fontsize',12,'fontname','Times New Roman','Interpreter', 'none')
% % %    
% % %     
% % %         
% % %         xlabel(h,'True Class','FontWeight','bold','fontsize',12,'fontname','Times New Roman')
% % %         ylabel(h,'Predicted Class','FontWeight','bold','fontsize',12,'fontname','Times New Roman','Interpreter', 'tex')
% % % 

%% Copy Confusion matrix to Figure(1) subplot
% % % 
% % % aH(1) = gca;
% % % acH(1) = {flipud(allchild(aH(1)))};
% % % ...........................................................
% % % figure(1)
% % %  p = get(gca, 'Position');
% % %  %sH = axes('Position', [p(1)+.3 p(2)+.2 p(3)-.4 p(4)-.4]); % 3 classes
% % %  sH = axes('Position', [p(1)+.20 p(2)+.10 p(3)-.25 p(4)-.20]); % 6 classes
% % % 
% % % ...........................................................
% % % for ii = 1:numel(acH{1})
% % %     copyobj(acH{1}(ii),sH(1));
% % % end
% % % set(sH,'Visible','on');
% % % set(sH,'Ydir','reverse')
% % %  
% % % 
% % %         if strcmp(name(end-3:end),'freq')==1
% % %             appendName = ', Self-powered Sensors';
% % %         else
% % %             appendName = ', MEMS Sensors';
% % %         end
% % % 
% % %         title(['Confusion Matrix for Case ',num2str(myCase),appendName],'FontWeight','bold', 'fontsize',12,'fontname','Times New Roman','Interpreter', 'none')
% % %         
% % %         xlabel(sH,'True Class','FontWeight','bold','fontsize',12,'fontname','Times New Roman')
% % %         ylabel(sH,'Predicted Class','FontWeight','bold','fontsize',12,'fontname','Times New Roman','Interpreter', 'tex')
% % % 
% % % 
% % % sH.XLabel.Visible = 'on'; 
% % % sH.YLabel.Visible = 'on'; 
% % % sH.Title.Visible  = 'on';
% % % 
% % % switch nClasses
% % %     case 2
% % % %     sH.XTickLabel = {' ',' ',' ','Intact',' ',['P',num2str(myCasesTable{myCase,3}(2))],' ',' ',' ',' '};
% % %     case 3
% % %     sH.XTickLabel = {' ','Intact',' ',['P',num2str(myCasesTable{myCase,3}(2))],' ',['P',num2str(myCasesTable{myCase,3}(3))],' ',' ',' ',' '};
% % %     case 5
% % %     sH.XTickLabel = {' ','Intact',['P',num2str(myCasesTable{myCase,3}(2))],['P',num2str(myCasesTable{myCase,3}(3))],['P',num2str(myCasesTable{myCase,3}(4))],...
% % %         ['P',num2str(myCasesTable{myCase,3}(5))],' ',' ',' '};
% % %     case 9
% % %     sH.XTickLabel = {' ','Intact',['P',num2str(myCasesTable{myCase,3}(2))],['P',num2str(myCasesTable{myCase,3}(3))],['P',num2str(myCasesTable{myCase,3}(4))],...
% % %         ['P',num2str(myCasesTable{myCase,3}(5))],['P',num2str(myCasesTable{myCase,3}(6))],['P',num2str(myCasesTable{myCase,3}(7))],['P',num2str(myCasesTable{myCase,3}(8))],['P',num2str(myCasesTable{myCase,3}(9))],' ',' '};
% % % end
% % %     sH.YTickLabel = sH.XTickLabel;
% % % 
% % % 
% % % sH.XRuler.Axle.LineStyle = 'none';  
% % % sH.YRuler.Axle.LineStyle = 'none';
% % % 
% % % sH.XTickLabelRotation = 0;
% % % sH.YTickLabelRotation = 90;

