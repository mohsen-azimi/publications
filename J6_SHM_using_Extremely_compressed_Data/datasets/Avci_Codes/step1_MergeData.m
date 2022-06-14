clc, clear, close all;

%
for pattern = [1:30]
    pattern
    fileID = fopen(['Dataset B/zzzBD',num2str(pattern),'.txt'], 'r');
    datacell = textscan(fileID, '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f',...
        'Delimiter', ',',...
        'Headerlines', 11,...
        'CollectOutput', 1) ;
    fclose(fileID);
    
    
    t = datacell{1, 1}(:,1);
    clear InputData0 TargetData0
    for sensor = 1:30
        x =  datacell{1, 1}(:,1+sensor) ;
        x = detrend(x);       % Remove Trend
        
        lenSignal = 1024;
        
        for n = 1: floor(numel(x)/lenSignal)
            InputData0(:,sensor,n) = x( (n-1)*lenSignal+1 : n*lenSignal);
            TargetData0(n,1) = pattern-1; %label
        end
    end
    
    
    
    %%  concatinate all; or save separately?
    
    if pattern==1
        InputData = InputData0;
        TargetData = TargetData0;
        
    else
        InputData = cat(3,InputData,InputData0);
        TargetData = [TargetData; TargetData0];
    end
    
    
    
    
    %% Save
    
end

nSamples = size(InputData,3);
nClasses = pattern;
nSensors = sensor;

clearvars -except t nSensors InputData TargetData lenSignal nClasses nSamples nSensors
%
save('Avci_B' )

beep
