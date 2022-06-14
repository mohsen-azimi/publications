%% Sensor specifications
function sensorSpec = f_sensSpecifications(dt)
%% Sensor #1: Accel
sensID = 1;  % first sensor
sensorSpec{sensID,1} = 'Accl_All';

sensorSpec{sensID,2}.nBins      = 1000;   % number of thresholds to record data
sensorSpec{sensID,2}.dt         = dt; % desired data for sampling (sensor specification)
sensorSpec{sensID,2}.minEdge    = -1.4;    % min threshold to be captured (default = 0)
sensorSpec{sensID,2}.maxEdge    = 1.4;   % max threshold to be captured (default = #)


              sensorSpec{sensID,2}.edge      = [linspace( sensorSpec{sensID,2}.minEdge,sensorSpec{sensID,2}.maxEdge, sensorSpec{sensID,2}.nBins+1)]';

                edgR= sensorSpec{sensID,2}.edge; edgR(1)=[]; edgL= sensorSpec{sensID,2}.edge; edgL(end)=[];
              sensorSpec{sensID,2}.binCntr=[edgL+(edgR-edgL)/2]; clear edgR edgL;

              sensorSpec{sensID,2}.binWidth = sensorSpec{sensID,2}.edge(2)-sensorSpec{sensID,2}.edge(1);

              
end             
%% clear 
% clear sensNum
