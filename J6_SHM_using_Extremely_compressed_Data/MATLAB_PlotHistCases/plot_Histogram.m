clear;  clc; close all;
t = [0:.001:10.23]';
        sensorSpec = f_sensSpecifications();

 
 figure(1)
   hold on; grid off; box on;
%    set(figure(1), 'Position', [100   100   883   400])
   set(figure(1), 'Position', [100   100   1000   500])

    ..................................................................................................
   load('C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\BenchMark\Outputs\Case1_Damage0_Seed001.mat') 
   subplot(3,3,1);
     plot(t, acc(:,16),'b','LineWidth',.5); xlim([min(t),max(t)])
     set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)                
     ylabel('Intact', 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

     annotation('arrow',[.35 .38],[.8 .8]);       annotation('arrow',[.63 .66],[.8 .8])


        [binCntr,freq, mu, sigma, scale] = f_TH2Hist(t,acc(:,16), sensorSpec,1); %#ok
        subplot(3,3,2);
        hist(acc(:,16),100)
        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
        line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
        xlim([-20,20])
        text(-18,350,['\mu = ',num2str(mu)],'HorizontalAlignment','left')
        text(+18,350,['\sigma = ',num2str(sigma)],'HorizontalAlignment','right')
        
            subplot(3,3,3);
            pd = makedist('Normal','mu',mu,'sigma',sigma);
            f_sensSpecifications();
            binWidth = sensorSpec{1,2}.binWidth;
            x_values = pd.mu-3*pd.sigma   :   binWidth/1   :   pd.mu+3*pd.sigma;
            y_values = pdf(pd,x_values)*scale;
            bar(x_values,y_values); hold on
            plot(x_values,y_values,'-b','linewidth',2); xlim([sensorSpec{1,2}.minEdge ,sensorSpec{1,2}.maxEdge])
            set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
            line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
..................................................................................................
   load('C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\BenchMark\Outputs\Case1_Damage1_Seed001.mat') 
   subplot(3,3,4);
    plot(t, acc(:,16),'b','LineWidth',.5); xlim([min(t),max(t)])
    set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)               
    ylabel('Damage Pattern 1', 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

    annotation('arrow',[.35 .38],[.5 .5]);       annotation('arrow',[.63 .66],[.5 .5])

        [binCntr,freq, mu, sigma, scale] = f_TH2Hist(t,acc(:,16), sensorSpec,1); %#ok
        subplot(3,3,5);
        hist(acc(:,16),100)
        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
        line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
        xlim([-20,20])
        text(-18,350,['\mu = ',num2str(mu)],'HorizontalAlignment','left')
        text(+18,350,['\sigma = ',num2str(sigma)],'HorizontalAlignment','right')

            subplot(3,3,6);
            pd = makedist('Normal','mu',mu,'sigma',sigma);
            f_sensSpecifications();
            binWidth = sensorSpec{1,2}.binWidth;
            x_values = pd.mu-3*pd.sigma   :   binWidth/1   :   pd.mu+3*pd.sigma;
            y_values = pdf(pd,x_values)*scale;
            bar(x_values,y_values); hold on
            plot(x_values,y_values,'-b','linewidth',2); xlim([sensorSpec{1,2}.minEdge ,sensorSpec{1,2}.maxEdge])
            set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
            line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')

..................................................................................................
   load('C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\BenchMark\Outputs\Case1_Damage2_Seed001.mat') 
   subplot(3,3,7);
   plot(t, acc(:,16),'b','LineWidth',.5); xlim([min(t),max(t)])
    set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)               
    ylabel('Damage Pattern 2', 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')
    xlabel({'Time-history (MEMS)';['(Reference domain with 10,000 data points)']}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

    annotation('arrow',[.35 .38],[.2 .2]);       annotation('arrow',[.63 .66],[.2 .2])

        [binCntr,freq, mu, sigma, scale] = f_TH2Hist(t,acc(:,16), sensorSpec,1); 
        subplot(3,3,8);
        hist(acc(:,16),100)
        set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
        line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')
        xlim([-20,20])
        text(-18,350,['\mu = ',num2str(mu)],'HorizontalAlignment','left')
        text(+18,350,['\sigma = ',num2str(sigma)],'HorizontalAlignment','right')
        xlabel({'Histogram (Self-powered) ';['( TL source domain with 100 data points)']}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')

            subplot(3,3,9);
            pd = makedist('Normal','mu',mu,'sigma',sigma);
            f_sensSpecifications();
            binWidth = sensorSpec{1,2}.binWidth;
            x_values = pd.mu-3*pd.sigma   :   binWidth/1   :   pd.mu+3*pd.sigma;
            y_values = pdf(pd,x_values)*scale;
            bar(x_values,y_values); hold on
            plot(x_values,y_values,'-b','linewidth',2); xlim([sensorSpec{1,2}.minEdge ,sensorSpec{1,2}.maxEdge])
            set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10) 
            xlabel({'Re-constructed histogram ';'(TL target domain with 3 data points, [\mu ,\sigma, {\itSF}])'}, 'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')
            line([mu;mu],[0;400],'LineWidth',1.5,'Color','k')

..................................................................................................   
   subplot(3,3,1)
% Leg=legend('IASC-ASCE SHM benchmark problem: Case #1, Sensor #16','Location','north','Orientation','horizontal');
Leg=legend('Data types for three different domains for the IASC-ASCE SHM benchmark problem','Location','north','Orientation','horizontal');
   pause(1)
 rect = [0.13, .94, .775, .05];
set(Leg, 'Position', rect,'fontsize',10, 'fontname','Times New Roman','FontWeight','Bold')
   
   
   
