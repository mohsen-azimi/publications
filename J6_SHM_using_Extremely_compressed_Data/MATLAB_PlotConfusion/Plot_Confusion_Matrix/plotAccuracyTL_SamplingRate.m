clear; close all; clc
%%
colors   = {'-r';   '-g';   '-b';   '-k';   '-m';    ':r';   ':g';    ':b'};
markers  = {'*';    's';    'o';     '+';    'x';     '+';    'x';    'x'};




for myCase = 1:1
    i = 0;
    for freq = [1 10 100 250 500 1000]
        i = i+1
files = {['A_SHM',num2str(myCase),'_freq2_SamplingFreq=',num2str(freq)]}; % 


%%
load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\Python\saveMATs\',files{1},'.mat']);




     scoreTableY(i,myCase) = scores(2);
     scoreTableX(i,myCase) = freq;

    end
    
    figure(1)
    plot(scoreTableX(:,1), 100*scoreTableY(:,myCase),[markers{myCase},colors{myCase}],'LineWidth',2,'MarkerSize', 6,'LineWidth',2, 'displayname',['Case',num2str(myCase)])
    pause(0.001)
    hold on
    
    ylim ([0 102])
    
    
   
end  


%%
 
set(figure(1), 'position', [200 200 700 300])


set(gca, 'LineWidth',1, 'FontWeight','normal', 'FontName','Times New Roman', 'FontSize',10)               



legend('show', 'location','southeast');
xlabel('Sampling Frequency (Hz)','FontWeight','bold','fontsize',12,'fontname','Times New Roman')
ylabel('Accuracy (%)','FontWeight','bold','fontsize',12,'fontname','Times New Roman','Interpreter', 'tex')





