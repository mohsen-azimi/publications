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

for myCase = 1:8
    
   
files = {['TL_A_SHM',num2str(myCase),'_freq1'];
         ['TL_A_SHM',num2str(myCase),'_freq2']}; % 
     
nClasses = numel(myCasesTable{myCase,3});

blueLine = {'--b';'-b'};
redLine  = {'--r';'-r'};

accLegends  = {'TL Source Domain, Training Accuracy';
               'TL Target Domain,  Training Accuracy '};
lossLegends = {'TL Source Domain, Training Loss';
               'TL Target Domain,  Training Loss'};

%%

for file = 1:2
 for fold = 1:10
     
    load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',files{file},'_fold',num2str(fold),'.mat']);
    Score(fold,1) = scores(2);
       
 end
     ScoresTable(myCase,file) = mean(Score);
     Err(myCase,file) = std(Score);
     


% % % % % figure(myCase); set(figure(myCase), 'position',[150         200        800         500])
% % % % % 
% % % % % yyaxis left;  
% % % % %         plot(AccuracyTH(1:min(500,numel(AccuracyTH)))*100,blueLine{file},'displayname',accLegends{file},'LineWidth',2); grid off; hold on
% % % % %         xlabel('Epoch','FontWeight','bold','fontsize',15,'fontname','Times New Roman')
% % % % %         ylabel('Accuracy (%)','FontWeight','bold','fontsize',15,'fontname','Times New Roman','Interpreter', 'tex')
% % % % % 
% % % % % pyyL = gca;
% % % % %       pyyL.YColor = [0 0 1];
% % % % %       
% % % % %       ylim([-0 100])
% % % % %       xlim([1 50])
% % % % % 
% % % % %         
% % % % % yyaxis right;  
% % % % %         plot(LossTH(1:min(500,numel(LossTH))),redLine{file},'displayname',lossLegends{file},'LineWidth',2); grid off; hold on
% % % % %         xlabel('Epoch','FontWeight','bold','fontsize',15,'fontname','Times New Roman')
% % % % %         ylabel('Loss','FontWeight','bold','fontsize',15,'fontname','Times New Roman','Interpreter', 'tex')
% % % % %         
% % % % % %         title([name,'_{',num,'}'],'FontWeight','bold','fontsize',18,'fontname','Times New Roman')
% % % % %  xlim([1 50])
% % % % % 
% % % % % pyyR = gca;
% % % % %       pyyR.YColor = [1 0 0];
% % % % %       pyyR.FontName = 'Times New Roman';
% % % % %       pyyR.FontSize = 15;
        
      
end

% % % legend('show', 'location','east');
% % % 
% % % % %    Title = {['{\bfCase} ',num2str(myCase)]};
% % % % %     X=xlim; Y=ylim;
% % % % %     text(.71*X(2),.65*Y(2),Title,...
% % % % %         'HorizontalAlignment','center','fontsize',14,...
% % % % %         'fontname','Times New Roman','FontWeight','Bold',...
% % % % %         'EdgeColor','w','BackgroundColor','w','LineWidth',1);
% % % % % 
% % % % %     grid on
    
    
    
    
   
end

%% Plot Test Accuracies

figure(55)
% set(figure(55), 'position', [200 200 800 300])
set(figure(55), 'position', [200 200 400 300]) % for two-cal paper
XTICK = categorical({'Case 1','Case 2','Case 3','Case 4','Case 5','Case 6','Case 7','Case 8'}); 

b = bar(XTICK,ScoresTable*100,0.4,'EdgeColor',[0 0 0],'LineWidth',1);
    hold on;
% b(1).FaceColor = 0.75*[1 1 1];  
% b(2).FaceColor = 0.25*[1 1 1];  
b(1).FaceColor = [0.3 0.3 1];  
b(2).FaceColor = [1 .3 .3];  

    
%         e = errorbar([[1:8]',[1:8]'],ScoresTable*100,Err);
%         set(e,'LineStyle', 'none');
   
ylim([80 102])

    set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)               
    ylabel('Test Accuracy (%)','FontWeight','normal','fontsize',10,'fontname','Times New Roman','Interpreter', 'tex')

legend('Source Domain', 'Target Domain', 'location','southeast')


b(1).BarWidth = 0.8;
b(2).BarWidth = 0.8;






