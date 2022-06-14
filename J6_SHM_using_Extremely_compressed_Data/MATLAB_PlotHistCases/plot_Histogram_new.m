clear;  clc; close all;
t = [0:.001:10.23]';
        sensorSpec = f_sensSpecifications();

        Colors = [.3 .3 1];
 
 figure(1); set(figure(1), 'Position', [100   10   400   800])
   load('C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\BenchMark\Outputs\Case1_Damage0_Seed001.mat') 


   subplot(3,1,1);
         plot(t, acc(:,16),'k','color',[.1 .1 1],'LineWidth',.5); xlim([min(t),max(t)])
         set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)                
         xlabel({'Time-domain Response History (MEMS)'}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')


        [binCntr,freq, mu, sigma, scale] = f_TH2Hist(t,acc(:,16), sensorSpec,1); 
    subplot(3,1,2);
        hist(acc(:,16),100);
        h = findobj(gca,'Type','patch');
        h.FaceColor = Colors;
        h.EdgeColor = 'k';
        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
        xlabel({'Discrete Histogram (Self-powered Sensor; Compressed)'}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

        line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
        xlim([-20,20])
        text(-18,350,['\mu = ',num2str(mu,3)],'HorizontalAlignment','left')
        text(+18,350,['\sigma = ',num2str(sigma,3)],'HorizontalAlignment','right')
        
    subplot(3,1,3);
        pd = makedist('Normal','mu',mu,'sigma',sigma);
        f_sensSpecifications();
        binWidth = sensorSpec{1,2}.binWidth;
        x_values = pd.mu-3*pd.sigma   :   binWidth/1   :   pd.mu+3*pd.sigma;
        y_values = pdf(pd,x_values)*scale;
        b = bar(x_values,y_values); hold on
        
        b(1).FaceColor = Colors; 

%         plot(x_values,y_values,'-b','color',.5*[1 1 1],'linewidth',2); xlim([sensorSpec{1,2}.minEdge ,sensorSpec{1,2}.maxEdge])
                text(-18,350,['\mu = ',num2str(mu,3)],'HorizontalAlignment','left')
                text(+18,350,['\sigma = ',num2str(sigma,3)],'HorizontalAlignment','right')
        xlim([-20,20])

        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
        line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
        xlabel({'Three-parameter Smooth Distribution'}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

        
        
%          annotation('arrow',[.52 .52],[.656 .63],'linewidth',15,'HeadWidth',40);       
%          annotation('arrow',[.52 .52],[.356 .33],'linewidth',15,'HeadWidth',40);       
         