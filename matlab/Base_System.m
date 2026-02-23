wirelessnetworkSupportPackageCheck
rng("default")                             % Reset the random number generator
numFrameSimulation = 5; 
% Simulation time in terms of number of 10 ms frames
networkSimulator = wirelessNetworkSimulator.init;
duplexType = "TDD";
gNBPosition = [0 0 30]; % [x y z] meters position in Cartesian coordinates
gNB = nrGNB(Position=gNBPosition, TransmitPower=60, SubcarrierSpacing=30000, ...
    CarrierFrequency=4.9e9, ChannelBandwidth=100e6, NumTransmitAntennas=32, NumReceiveAntennas=32, ...
    DuplexMode=duplexType, ReceiveGain=11, SRSPeriodicityUE=20, NumResourceBlocks=273);
csiMeasurementSignalDLType = "CSI-RS";
muMIMOConfiguration = struct(MaxNumUsersPaired=2, MaxNumLayers=8, MinNumRBs=2, MinSINR=10);
allocationType = 0; 
% Resource allocation type
configureScheduler(gNB, ResourceAllocationType=allocationType, MaxNumUsersPerTTI=10, ...
    MUMIMOConfigDL=muMIMOConfiguration, CSIMeasurementSignalDL=csiMeasurementSignalDLType);
numUEs =10; 
% Number of UE nodes
% For all UEs, specify position in spherical coordinates (r, azimuth, elevation)
% relative to gNB.
ueRelPosition = [ones(numUEs, 1)*500 (rand(numUEs, 1)-0.5)*120 zeros(numUEs, 1)];
% Convert spherical to Cartesian coordinates considering gNB position as origin
[xPos, yPos, zPos] = sph2cart(deg2rad(ueRelPosition(:, 2)), deg2rad(ueRelPosition(:, 3)), ...
    ueRelPosition(:, 1));
% Convert to absolute Cartesian coordinates
uePositions = [xPos yPos zPos] + gNBPosition;
ueNames = "UE-" + (1:size(uePositions, 1));
UEs = nrUE(Name=ueNames, Position=uePositions, ReceiveGain=0, NumTransmitAntennas=1, NumReceiveAntennas=1);
connectUE(gNB, UEs, FullBufferTraffic="DL", CSIReportPeriodicity=10)
addNodes(networkSimulator, gNB)
addNodes(networkSimulator, UEs)
% Set delay profile, delay spread (in seconds), and maximum Doppler
% shift (in Hz)
channelConfig = struct(DelayProfile="CDL-C", DelaySpread=10e-9, MaximumDopplerShift=1);
channels = hNRCreateCDLChannels(channelConfig, gNB, UEs);
customChannelModel  = hNRCustomChannelModel(channels);
addChannelModel(networkSimulator, @customChannelModel.applyChannelModel)
enableTraces = true;
if enableTraces
    % Create an object for scheduler trace logging
    simSchedulingLogger = helperNRSchedulingLogger(numFrameSimulation, gNB, UEs);
    % Create an object for PHY trace logging
    simPhyLogger = helperNRPhyLogger(numFrameSimulation, gNB, UEs);
end
% This parameter impacts the simulation time
numMetricPlotUpdates =1000;
metricsVisualizer = helperNRMetricsVisualizer(gNB, UEs, RefreshRate=numMetricPlotUpdates, ...
    PlotSchedulerMetrics=true, PlotPhyMetrics=false, PlotCDFMetrics=true, LinkDirection=0);
simulationLogFile = "simulationLogs"; % For logging the simulation traces
% Calculate the simulation duration (in seconds)
simulationTime = numFrameSimulation*1e-2;
% Run the simulation
run(networkSimulator, simulationTime);
% Read performance metrics
displayPerformanceIndicators(metricsVisualizer)
if enableTraces
    simulationLogs = cell(1, 1);
    if gNB.DuplexMode == "FDD"
        logInfo = struct(DLTimeStepLogs=[], ULTimeStepLogs=[], ...
            SchedulingAssignmentLogs=[], PhyReceptionLogs=[]);
        [logInfo.DLTimeStepLogs, logInfo.ULTimeStepLogs] = getSchedulingLogs(simSchedulingLogger);
    else % TDD
        logInfo = struct(TimeStepLogs=[], SchedulingAssignmentLogs=[], PhyReceptionLogs=[]);
        logInfo.TimeStepLogs = getSchedulingLogs(simSchedulingLogger);
    end
    % Get the scheduling assignments log
    logInfo.SchedulingAssignmentLogs = getGrantLogs(simSchedulingLogger);
    % Get the PHY reception logs
    logInfo.PhyReceptionLogs = getReceptionLogs(simPhyLogger);
    % Save simulation logs in a MAT-file
    simulationLogs{1} = logInfo;
    save(simulationLogFile, "simulationLogs")
end
if enableTraces
    avgNumUEsPerRB = calculateAvgUEsPerRBDL(logInfo, gNB.NumResourceBlocks, allocationType, duplexType);
    % Plotting the histogram
    figure;
    theme("light")
    histogram(avgNumUEsPerRB, 'BinWidth', 0.1);
    title('Distribution of Average Number of UEs per RB in DL Slots');
    xlabel('Average Number of UEs per RB');
    ylabel('Number of Occurrence');
    grid on;
end
function avgUEsPerRB = calculateAvgUEsPerRBDL(logInfo, numResourceBlocks, ratType, duplexMode)
    %calculateAvgUEsPerRBDL Calculate average number of UE nodes per RB in DL slots.
    %   AVGUESPERRB = calculateAvgUEsPerRBDL(LOGINFO, NUMRESOURCEBLOCKS, RATTYPE, DUPLEXMODE)
    %   calculates the average number of UE nodes per RB for each DL slot based on the log information.
    %
    %   LOGINFO is a structure containing detailed logs of the simulation.
    %   NUMRESOURCEBLOCKS is an integer specifying the number of resource blocks in carrier bandwidth
    %   RATTYPE is an integer indicating the resource allocation type (0 or 1).
    %   DUPLEXMODE is a string that indicates the duplex mode, which can be either "TDD" or "FDD".
    %
    %   The function returns AVGUEsPerRB, a vector containing the average number of
    %   UE nodes per RB for each DL slot, allowing for analysis of resource allocation efficiency.
    
    % Determine the appropriate data source based on the duplex mode
    if strcmp(duplexMode, 'TDD')
        timeStepLogs = logInfo.TimeStepLogs;
        freqAllocations = timeStepLogs(:, 5);
    elseif strcmp(duplexMode, 'FDD')
        timeStepLogs = logInfo.DLTimeStepLogs;
        freqAllocations = timeStepLogs(:, 4);
    end
    
    % Extract the number of slots
    numOfSlots = size(timeStepLogs, 1) - 1;
    
    % Determine RBG sizes for RAT0
    if ~ratType
        % Determine the nominal RBG size P based on TR 38.214
        numRBG = size(freqAllocations{2}, 2);
        P = ceil(numResourceBlocks / numRBG);
        % Initialize the RBG sizes
        numRBsPerRBG = P * ones(1, numRBG);
        % Calculate the size of the last RBG
        remainder = mod(numResourceBlocks, P);
        if remainder > 0
            numRBsPerRBG(end) = remainder;
        end
    end
    
    % Initialize a vector to store the average number of UEs per RB for DL slots
    avgUEsPerRB = zeros(1, numOfSlots);
    
    % Iterate over each slot
    for slotIdx = 1:numOfSlots
        % For TDD, extract the type of slot (DL or UL)
        if strcmp(duplexMode, 'TDD')
            slotType = timeStepLogs{slotIdx + 1, 4};
            if ~strcmp(slotType, 'DL')
                continue; % Skip UL slots in TDD
            end
        end
    
        % Extract the frequency allocation for the current DL slot
        freqAllocation = freqAllocations{slotIdx + 1};
    
        if ~ratType
            % Process RAT0
            totalUniqueUEs = sum(arrayfun(@(rbgIdx) nnz(freqAllocation(:, rbgIdx) > 0) * ...
                numRBsPerRBG(rbgIdx), 1:length(numRBsPerRBG)));
            avgUEsPerRB(slotIdx) = totalUniqueUEs / numResourceBlocks;
    
        else
            % Process RAT1
            ueRBUsage = zeros(1, numResourceBlocks); % Vector to count UEs per RB
    
            for ueIdx = 1:size(freqAllocation, 1)
                startRB = freqAllocation(ueIdx, 1);
                numContiguousRBs = freqAllocation(ueIdx, 2);
                ueRBUsage(startRB + 1:(startRB + numContiguousRBs)) = ...
                    ueRBUsage(startRB + 1:(startRB + numContiguousRBs)) + 1;
            end
    
            avgUEsPerRB(slotIdx) = mean(ueRBUsage(ueRBUsage > 0));
        end
    end
    
    % Remove entries for UL slots (if any), which is relevant for TDD
    avgUEsPerRB = avgUEsPerRB(avgUEsPerRB > 0);
end