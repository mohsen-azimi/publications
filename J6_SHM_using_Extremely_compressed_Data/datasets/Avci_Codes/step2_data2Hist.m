%% Load data
clear; clc; close all
%% Read the result data 

load('Avci_B.mat');



    
    
...====================================================================
...                          TH2Hist
...====================================================================
dt = .001;
sensorSpec = f_sensSpecifications(dt);
sensID     = 1;

% t = t;
binCntr = zeros(sensorSpec{1, 2}.nBins,nSensors,nSamples);
binWidth = sensorSpec{1,2}.binWidth;
freq    = zeros(sensorSpec{1, 2}.nBins,nSensors,nSamples);
freq2   = freq;
mu      = zeros(1,nSensors,nSamples);
sigma   = zeros(1,nSensors,nSamples);


for sample = 1:nSamples     % #Samples
    clc, 
    sample
    for sens = 1:nSensors   % #Sensors
    signal = InputData(:,sens,sample);
    [binCntr(:,sens,sample),freq(:,sens,sample), mu(1,sens,sample), sigma(1,sens,sample), scale(1,sens,sample)] = f_TH2Hist(t,signal, sensorSpec,sensID);
    
    pd = makedist('Normal','mu',mu(1,sens,sample),'sigma',sigma(1,sens,sample));
    
    freq2(:,sens,sample) = pdf(pd,binCntr(:,sens,sample))*scale(1,sens,sample);
      
%     figure(1)
%     plot(binCntr(:,sens,sample),freq2(:,sens,sample),'-b','linewidth',2); xlim([sensorSpec{1,2}.minEdge ,sensorSpec{1,2}.maxEdge])
%     hold on
% 
%     plot(binCntr(:,sens,sample),freq(:,sens,sample),'-r','linewidth',.6)
%     hold off
%     pause(0.01)

    end

end
...====================================================================
InputData  = freq;
InputData2 = freq2; % regenerated using mu/sigma/scale/+binCntr


lenSignal  = size(freq,1);    
...====================================================================
%% Divide into 70%-30% for TL (70 sourceDomain -- 30 targetDomain)

 TL_sourceIndx = 1:nSamples;
 
 TL_targetIndx = randperm(nSamples, round(  0.3*nSamples));
 TL_sourceIndx(TL_targetIndx) = []; 
 

 TL_Source_InputData = freq(:,:,TL_sourceIndx);
 TL_Target_InputData = freq2(:,:,TL_targetIndx); 

 TL_Source_TargetData = TargetData(TL_sourceIndx);
 TL_Target_TargetData = TargetData(TL_targetIndx); 
 
 TL_Source_nSamples = numel(TL_sourceIndx);
 TL_Target_nSamples = numel(TL_targetIndx);



%%

save('Avci_B_freq',...
    'InputData', 'InputData2', 'TL_Source_InputData','TL_Target_InputData',...
    'TargetData','TL_Source_TargetData','TL_Target_TargetData',...
    'TL_Source_nSamples', 'TL_Target_nSamples',...
    'nSamples','lenSignal','nSensors','nClasses' )
.......................................................................





disp('Done!')
beep
