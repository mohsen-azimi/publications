% clear; close all;
 function [binCntr, freq, mu, sigma, scale] = f_TH2Hist(time,TH, sensorSpec,sensID)
%% Sensor specifications

nBins      = sensorSpec{sensID,2}.nBins; % predefined number of gates 
minEdge    = sensorSpec{sensID,2}.minEdge;   
maxEdge    = sensorSpec{sensID,2}.maxEdge;   

% %               sensorSpec{sensID,2}.edge      = linspace( sensorSpec{sensID,2}.minEdge,sensorSpec{sensID,2}.maxEdge, sensorSpec{sensID,2}.nBins+1);
% %                 edgR= sensorSpec{sensID,2}.edge; edgR(1)=[]; edgL= sensorSpec{sensID,2}.edge; edgL(end)=[];
% %               sensorSpec{sensID,2}.binCntr=edgL+(edgR-edgL)/2; clear edgR edgL;
% % 
% %               sensorSpec{sensID,2}.binWidth = sensorSpec{sensID,2}.edge(2)-sensorSpec{sensID,2}.edge(1);

edge      = sensorSpec{sensID,2}.edge;
binCntr   = sensorSpec{sensID,2}.binCntr;
binWidth  = sensorSpec{sensID,2}.binWidth;

%% Modify TH according to the new dt for the sensor
% % %        
% % % dtHist = sensorSpec{sensID,2}.dt; 
% % % tHist  = [0:dtHist:time(end)]';
% % % TH_orig     = interp1(time,TH,tHist); % time-history revised according to sensor sampling
% % % 
% % % TH = (TH_orig(:));  % keep the original for later use
% % % %  TH = TH_orig(:) ! not completed, revise edges;      % If negative data points are important
%% 
% close all;

TH(isnan(TH)) = [];     % Remove NaN elements
TH(abs(TH)<minEdge)=[];      % Apply the first/last threshold; 
TH(abs(TH)>maxEdge)=maxEdge    .* abs(TH(abs(TH)>maxEdge))   ./   (TH(abs(TH)>maxEdge)); % Apply the first/last threshold; The last term in to keep signs



     
                [freq,~]      = histcounts(TH,edge);
%                 edgR= edge; edgR(1)=[]; edgL= edge; edgL(end)=[];
%                 binCntr=edgL+(edgR-edgL)/2; clear edgR edgL;
% 
%                 binWidth = binCntr(end)-binCntr(1);
%                 edgeSize = edge(2)-edge(1);
               
                freq        = freq(:);  % column vector
                
                if sum(freq>0)<3
                    beep, pause(.0001), beep
                end
% % %                  freqNrmlizd = freq/trapz(binCntr,freq); %  let's normalize the histogram (using Trapezoid rule)
% % % 
% % %                   for i=1:numel(edge)
% % %                     hold on
% % %                     plot([edge(i),edge(i)],[0,max(freq)],'Color','c') % to show the edges only 
% % %                   end
                  
%                  histfit(TH,sum(freq>0));  % sum(freq>0) = nBins (remove zeros)
% % % % %                 figure(100); subplot(1,2,1)
% % % % %                 pause(0.00001);
% % % % % %                 frame_h = get(handle(gcf),'JavaFrame');
% % % % % %                 set(frame_h,'Maximized',1); 
% % % % % 
% % % % % %                  subplot(2,2,sensID);    % clear for the next plot           
% % % % % %                  hold off
% % % % % %                  bar(binCntr, freq,'c');    %%  plot( binCntr, freq,'*r'); 
% % % % %                  hold on
% % % % %                  xlabel('Gates (thresholds)'); ylabel('Frequency'); title(['Histogram for  ',sensorSpec{sensID,2}.Location])
% % % % %                  xlim([minEdge,maxEdge])
                 
                 


% % % histfit(TH,nBins,'normal')  % plot MATLAB histogram
pd = fitdist(binCntr,'normal','frequency',freq);
x_values = pd.mu-3*pd.sigma:binWidth/100:pd.mu+3*pd.sigma;%min(TH_orig):binWidth/100:max(TH_orig);
y_valuse = pdf(pd,x_values);
% % % % hold on, plot(x_values,trapz(binCntr,freq)*y_valuse,'-k')


mu  = pd.mu;
sigma = pd.sigma;
scale = trapz(binCntr,freq);
% pause(0.0000000001)

 end
     