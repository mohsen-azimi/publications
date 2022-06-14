clear; clc; close all
%% load Data 
file = 'A_SHM2_freq2.mat';

% load(['C:\MOHSEN\Research\Temp\03_CNN\AISC_BenchMark\BenchMark\Outputs\',file]);


% for i = 1:nSamples
%     
%     x(i,:) = [InputData2(:,1,i)]';
%     
% end

% % % plot(x');
% % %     
% % % [u,I,J] = unique(x, 'rows', 'first');
% % % hasDuplicates = size(u,1) < size(x,1)
% % % ixDupRows = setdiff(1:size(x,1), I)
% % % dupRowValues = x(ixDupRows,:)
% % % 


%% Experimental SHM
clear
file = 'allTHData_shm_a_freq.mat';
% load(['C:\MOHSEN\Research\Benchmarks\ERA Identification SHM Benchmark\NEES Files\files_bldg_shm_exp2\CD of UBC.experiment 2002 2\data\Ambient\Ambient\',file]);
% 
% 
% for i = 181:350
%     
%     x(i,:) = [InputData2(:,1,i)]';
%     
% end
% 
% 
% plot(x');
%% Avci
clear

file = 'Avci_B_freq.mat';
load(['C:\MOHSEN\Research\Temp\03_CNN\Dataset\Ovci\',file]);


clear x; ii=0; close all
for i = [256*1+1:256*1+50]%, 256*15+1:256*15+50]
    ii = ii+1;
    x(ii,:) = [InputData2(:,1,i)]';
end


plot(x','r');
hold on


clear x; ii=0;
for i = [256*15+1:256*15+50]
    ii = ii+1;
    xx(ii,:) = [InputData2(:,15,i)]';
end


plot(xx','b');
hold on

