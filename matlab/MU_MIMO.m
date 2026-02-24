%% ====================== 1. SYSTEM CONFIGURATION ======================

% --- gNB CONFIGURATION (32T32R) ---
gNBConfig = struct();
gNBConfig.Position = [0 0 30];            % Vị trí [x,y,z]
gNBConfig.TransmitPower = 60;             % Công suất phát (dBm) ~ theo yêu cầu trung bình
gNBConfig.SubcarrierSpacing = 30000;      % 30 kHz
gNBConfig.CarrierFrequency = 4.9e9;       % 4.9 GHz
gNBConfig.ChannelBandwidth = 100e6;       % 100 MHz
% Với 100MHz @ 30kHz SCS, số lượng RB chuẩn là 273
gNBConfig.NumResourceBlocks = 273;        
gNBConfig.NumTransmitAntennas = 32;       % 32 Anten phát
gNBConfig.NumReceiveAntennas = 32;        % 32 Anten thu
gNBConfig.ReceiveGain = 32;               % Gain thu 32 dBi
gNBConfig.DuplexMode = "TDD";             % TDD
gNBConfig.SRSPeriodicity = 5;             % SRS Periodicity = 5 slots

% --- UE CONFIGURATION ---
ueConfig = struct();
ueConfig.NumUEs = 16;                    % 16 UEs connected
ueConfig.NumTransmitAntennas = 4;
ueConfig.NumReceiveAntennas = 4;          % 4 Anten thu mỗi UE
ueConfig.ReceiveGain = 0;                 % Gain 0 dBi
ueConfig.MaxDistance = 1200;              % Bán kính 1200m
ueConfig.MinDistance = 10;                
ueConfig.AzimuthRange = [-30 30];         % Góc phương vị +/- 30 độ
ueConfig.ElevationAngle = 0;    
ueConfig.NoiseFigureMin = 7;              % More realistic NF: 7-9 dB
ueConfig.NoiseFigureMax = 9;             

% --- MU-MIMO & SCHEDULER ---
muMIMOConfig = struct();
muMIMOConfig.MaxNumUsersPaired = 4;       % 
muMIMOConfig.MinNumRBs = 3;               % Min 3 RBs
muMIMOConfig.SemiOrthogonalityFactor = 0.6; 
muMIMOConfig.MinCQI = 1;                  % Tương ứng CQI 1
muMIMOConfig.MaxNumLayers = 16;          % 16 spatial streams per TTI (matches Python --n_layers 16)
schedulerConfig = struct();
schedulerConfig.ResourceAllocationType = 0;  % RB-based
schedulerConfig.MaxNumUsersPerTTI = 10;      % Max 4 UE scheduled per TTI
schedulerConfig.SignalType = "CSI-RS";      % Use CSI-RS for channel measurement         

% --- CSI report granularity (Subband/PRG PMI) ---
% Choose subband/PRG size in RBs. Typical values: 2 or 4.
csiReportConfig = struct();
csiReportConfig.SubbandSize = 16;
csiReportConfig.PRGSize = 4;

% --- CHANNEL MODEL ---
channelConfig = struct();
channelConfig.DelayProfile = "CDL-D";       
channelConfig.DelaySpread = 450e-9;         % 450ns
channelConfig.MaxDopplerShift = 136;        % ~136 Hz
channelConfig.Orientation = [60; 0; 0];     % Hướng anten gNB

% --- SIMULATION CONTROL ---
simConfig = struct();
simConfig.NumFrameSimulation = 5;          
simConfig.EnableTraces = true;              

%% ====================== 2. INITIALIZATION ======================

wirelessnetworkSupportPackageCheck
rng("default");
networkSimulator = wirelessNetworkSimulator.init;

% --- Create gNB (32T32R) ---
gNB = nrGNB('Position', gNBConfig.Position, ...
    'TransmitPower', gNBConfig.TransmitPower, ...
    'SubcarrierSpacing', gNBConfig.SubcarrierSpacing, ...
    'CarrierFrequency', gNBConfig.CarrierFrequency, ...
    'ChannelBandwidth', gNBConfig.ChannelBandwidth, ...
    'NumTransmitAntennas', gNBConfig.NumTransmitAntennas, ... 
    'NumReceiveAntennas', gNBConfig.NumReceiveAntennas, ...   
    'DuplexMode', gNBConfig.DuplexMode, ...
    'ReceiveGain', gNBConfig.ReceiveGain, ...
    'SRSPeriodicityUE', gNBConfig.SRSPeriodicity, ... 
    'NumResourceBlocks', gNBConfig.NumResourceBlocks);

% --- Explicit CSI-RS configuration (ensure CSI-RS is enabled for UEs) ---
% Create CSI-RS config manually (same logic as gNB.createCSIRSConfiguration)
csirsRowNumberTable = [
    1 1 1; % Row 1: density 3 (not used)
    1 1 1; % Row 2
    2 1 1; % Row 3
    4 1 1; % Row 4
    4 1 1; % Row 5
    8 4 1; % Row 6
    8 2 1; % Row 7
    8 2 1; % Row 8
    12 6 1; % Row 9
    12 3 1; % Row 10
    16 4 1; % Row 11
    16 4 1; % Row 12
    24 3 2; % Row 13
    24 3 2; % Row 14
    24 3 1; % Row 15
    32 4 2; % Row 16
    32 4 2; % Row 17
    32 4 1; % Row 18
];
subcarrierSet = [1 3 5 7 9 11]; % k0 k1 k2 k3 k4 k5
symbolSet = [0 4]; % l0 l1

csirsCfg = nrCSIRSConfig(CSIRSType="nzp", NumRB=gNBConfig.NumResourceBlocks);
rowNum = find(csirsRowNumberTable(2:end, 1) == gNBConfig.NumTransmitAntennas, 1) + 1;
csirsCfg.RowNumber = rowNum;
csirsCfg.SubcarrierLocations = subcarrierSet(1:csirsRowNumberTable(rowNum, 2));
csirsCfg.SymbolLocations = symbolSet(1:csirsRowNumberTable(rowNum, 3));
% CSI staleness fix: With Doppler 136Hz, coherence time ~3.7ms
% CSI period must be << coherence time for accurate precoding
csirsCfg.CSIRSPeriod = [10 0]; % Reduced from 10 to 2 slots (~1ms)
gNB.CSIRSConfiguration = csirsCfg;
fprintf('CSI-RS configured: %d ports, Row %d, Period [%d %d]\n', ...
    gNBConfig.NumTransmitAntennas, rowNum, csirsCfg.CSIRSPeriod(1), csirsCfg.CSIRSPeriod(2));

% --- Configure Scheduler ---
if schedulerConfig.SignalType == "SRS"
    muMIMOStruct = struct(...
        'MaxNumUsersPaired', muMIMOConfig.MaxNumUsersPaired, ...
        'MinNumRBs', muMIMOConfig.MinNumRBs, ...
        'MaxNumLayers', muMIMOConfig.MaxNumLayers);
else
    muMIMOStruct = struct(...
        'MaxNumUsersPaired', muMIMOConfig.MaxNumUsersPaired, ...
        'MinNumRBs', muMIMOConfig.MinNumRBs, ...
        'SemiOrthogonalityFactor', muMIMOConfig.SemiOrthogonalityFactor, ...
        'MinCQI', muMIMOConfig.MinCQI, ...
        'MaxNumLayers', muMIMOConfig.MaxNumLayers);
end

% === DRL Scheduler (nrDRLScheduler với Socket) ===
drlScheduler = nrDRLScheduler();

% Cấu hình DRL scheduler
drlScheduler.EnableDRL = true;
drlScheduler.TrainingMode = true;  % Enable layer-by-layer training protocol (train_matlab.py)
drlScheduler.NumLayers = muMIMOConfig.MaxNumLayers;  % 16 layers
drlScheduler.MaxUsersPerRBG = muMIMOConfig.MaxNumUsersPaired;
drlScheduler.MaxUEs = ueConfig.NumUEs;  % Số UE tối đa (cho feature matrix)
drlScheduler.SubbandSize = csiReportConfig.SubbandSize;  % Subband size cho CQI features
drlScheduler.DRLDebug = true;  % Bật debug output

% CSI-RS based MU-MIMO constraints
drlScheduler.EnableI1Constraint = true;              % UEs must have same i1 (wideband beam)
drlScheduler.EnableOrthogonalityConstraint = true;   % Check precoder orthogonality
drlScheduler.SemiOrthogonalityFactor = muMIMOConfig.SemiOrthogonalityFactor;  % 0.7
drlScheduler.MU_MCSBackoff = 2;                      % Reduce MCS by 2 per co-scheduled UE

% Kết nối tới Python DRL server qua socket
ok = drlScheduler.connectToDRLAgent('127.0.0.1', 5555);
assert(ok, 'Cannot connect to DRL server. Start python train_drl_with_matlab.py first.');

% --- Link Adaptation (OLLA) for MU-MIMO ---
% BLER was 40-60%, need more aggressive OLLA to converge faster
% Target BLER ~10% (0.1): StepUp/StepDown ratio should be ~9
linkAdaptationConfig = struct(...
    'InitialOffset', 6, ...   % Reduced from 12 (start more conservative)
    'StepUp', 0.5, ...        % Reduced from 1.0 (increase MCS slower on ACK)
    'StepDown', 0.1);         % Increased from 0.02 (decrease MCS faster on NACK)       

configureScheduler(gNB, ...
    'Scheduler', drlScheduler, ...
    'ResourceAllocationType', schedulerConfig.ResourceAllocationType, ... 
    'MaxNumUsersPerTTI', schedulerConfig.MaxNumUsersPerTTI, ...
    'MUMIMOConfigDL', muMIMOStruct, ...
    'LinkAdaptationConfigDL', linkAdaptationConfig, ...
    'CSIMeasurementSignalDL', schedulerConfig.SignalType);

% --- Create UEs ---
UEs = nrUE.empty(0, ueConfig.NumUEs); 
rng(42); 

ueAzimuths = ueConfig.AzimuthRange(1) + (ueConfig.AzimuthRange(2) - ueConfig.AzimuthRange(1)) * rand(ueConfig.NumUEs, 1);
ueElevations = zeros(ueConfig.NumUEs, 1);
ueDistances = ueConfig.MinDistance + (ueConfig.MaxDistance - ueConfig.MinDistance) * rand(ueConfig.NumUEs, 1);

[xPos, yPos, zPos] = sph2cart(deg2rad(ueAzimuths), deg2rad(ueElevations), ueDistances);
uePositions = [xPos yPos zPos] + gNBConfig.Position;

fprintf('Khoi tao %d UEs (gNB: 32T32R, Ptx: %ddBm)...\n', ueConfig.NumUEs, gNBConfig.TransmitPower);

for i = 1:ueConfig.NumUEs
    currentNoise = ueConfig.NoiseFigureMin + (ueConfig.NoiseFigureMax - ueConfig.NoiseFigureMin) * rand();
    UEs(i) = nrUE('Name', "UE-" + string(i), ...
                  'Position', uePositions(i, :), ...
                  'NumReceiveAntennas', ueConfig.NumReceiveAntennas, ... 
                  'NoiseFigure', currentNoise, ... 
                  'ReceiveGain', ueConfig.ReceiveGain, ...
                  'NumTransmitAntennas',ueConfig.NumTransmitAntennas);           
end

connectUE(gNB, UEs, FullBufferTraffic="on", ...
    CSIReportPeriodicity=10, CSIRSConfig=gNB.CSIRSConfiguration, ...  % Reduced to match CSI-RS period
    CustomContext=csiReportConfig);

% % === MIXED TRAFFIC SCENARIO ===
% % UE 1-6: Video streaming (high data rate, bursty)
% for i = 1:6
%     traffic = networkTrafficVideoConference('HasJitter', true);
%     addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
% end
% 
% % UE 7-10: VoIP (low latency, small packets)
% for i = 7:10
%     traffic = networkTrafficVoIP;
%     addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
% end
% 
% % UE 11-13: FTP (bursty file transfer)
% for i = 11:13
%     traffic = networkTrafficFTP;
%     addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
% end
% 
% % UE 14-16: On-Off Application (web browsing pattern)
% for i = 14:16
%     traffic = networkTrafficOnOff( ...
%         'OnTime', 0.02, ...           % 20ms ON
%         'OffTime', 0.1, ...           % 100ms OFF  
%         'DataRate', 5e6, ...          % 5 Mbps when ON
%         'PacketSize', 1500);          % 1500 bytes/packet
%     addTrafficSource(gNB, traffic, 'DestinationNode', UEs(i));
% end
% 
% fprintf('Traffic configured: UE1-6 Video, UE7-10 VoIP, UE11-13 FTP, UE14-16 OnOff\n');

addNodes(networkSimulator, gNB);
addNodes(networkSimulator, UEs);

%% ====================== 3. CHANNEL MODEL ======================

cdlConfig = struct(...
    'DelayProfile', channelConfig.DelayProfile, ...
    'DelaySpread', channelConfig.DelaySpread, ...
    'MaximumDopplerShift', channelConfig.MaxDopplerShift, ...
    'TransmitArrayOrientation', channelConfig.Orientation);

% Tạo kênh truyền CDL
channels = hNRCreateCDLChannels(cdlConfig, gNB, UEs);
customChannelModel  = hNRCustomChannelModel(channels);
addChannelModel(networkSimulator, @customChannelModel.applyChannelModel);

%% ====================== 4. RUN SIMULATION ======================

if simConfig.EnableTraces
    simSchedulingLogger = helperNRSchedulingLogger(simConfig.NumFrameSimulation, gNB, UEs);
    simPhyLogger = helperNRPhyLogger(simConfig.NumFrameSimulation, gNB, UEs);
end

% Visualizer
metricsVisualizer = helperNRMetricsVisualizer(gNB, UEs, ...
    'RefreshRate', 1000, ... 
    'PlotSchedulerMetrics', true, ...
    'PlotPhyMetrics', false, ...
    'PlotCDFMetrics', true, ...
    'LinkDirection', 0);

simulationLogFile = "simulationLogs_128T128R"; 
simulationTime = simConfig.NumFrameSimulation * 1e-2;

fprintf('Dang chay mo phong 128T128R trong %.2f giay...\n', simulationTime);
run(networkSimulator, simulationTime);

%% ====================== 5. LOGS & METRICS ======================
displayPerformanceIndicators(metricsVisualizer);

if simConfig.EnableTraces
    simulationLogs = cell(1, 1);
    % Logic lấy log TDD/FDD
    if gNB.DuplexMode == "FDD"
        logInfo = struct('DLTimeStepLogs',[], 'ULTimeStepLogs',[], 'SchedulingAssignmentLogs',[], 'PhyReceptionLogs',[]);
        [logInfo.DLTimeStepLogs, logInfo.ULTimeStepLogs] = getSchedulingLogs(simSchedulingLogger);
    else 
        logInfo = struct('TimeStepLogs',[], 'SchedulingAssignmentLogs',[], 'PhyReceptionLogs',[]);
        logInfo.TimeStepLogs = getSchedulingLogs(simSchedulingLogger);
    end
    
    logInfo.SchedulingAssignmentLogs = getGrantLogs(simSchedulingLogger);
    logInfo.PhyReceptionLogs = getReceptionLogs(simPhyLogger);
    save(simulationLogFile, "simulationLogs");
    
    % Plot Histrogram UE/RB
    avgNumUEsPerRB = calculateAvgUEsPerRBDL(logInfo, gNB.NumResourceBlocks, ...
        schedulerConfig.ResourceAllocationType, gNBConfig.DuplexMode);
    
    figure; theme("light");
    histogram(avgNumUEsPerRB, 'BinWidth', 0.1);
    title('Distribution of Avg UEs per RB (128T128R Configuration)');
    xlabel('Average Number of UEs per RB');
    ylabel('Frequency');
    grid on;
    
    % Plot UE Pairing Statistics (how many UEs paired per RBG)
    [pairingCounts, pairingDistribution] = calculateUEPairingStats(logInfo, ...
        gNB.NumResourceBlocks, schedulerConfig.ResourceAllocationType, gNBConfig.DuplexMode);
    
    figure; theme("light");
    bar(pairingCounts(:,1), pairingCounts(:,2));
    title('UE Pairing Distribution per RBG');
    xlabel('Number of UEs Paired on RBG');
    ylabel('Count (RBG-slot occurrences)');
    xticks(0:max(pairingCounts(:,1)));
    grid on;
    
    % Add text labels on bars
    for i = 1:size(pairingCounts, 1)
        text(pairingCounts(i,1), pairingCounts(i,2), num2str(pairingCounts(i,2)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    % Print summary
    fprintf('\n=== UE PAIRING STATISTICS ===\n');
    totalRBGSlots = sum(pairingCounts(:,2));
    for i = 1:size(pairingCounts, 1)
        numUEs = pairingCounts(i,1);
        count = pairingCounts(i,2);
        pct = 100 * count / totalRBGSlots;
        if numUEs == 0
            fprintf('  Empty RBGs: %d (%.1f%%)\n', count, pct);
        elseif numUEs == 1
            fprintf('  Single UE (no pairing): %d (%.1f%%)\n', count, pct);
        else
            fprintf('  %d UEs paired: %d (%.1f%%)\n', numUEs, count, pct);
        end
    end
end

%% ====================== HELPER FUNCTION ======================
function avgUEsPerRB = calculateAvgUEsPerRBDL(logInfo, numResourceBlocks, ratType, duplexMode)
    if strcmp(duplexMode, 'TDD')
        timeStepLogs = logInfo.TimeStepLogs;
        freqAllocations = timeStepLogs(:, 5);
    elseif strcmp(duplexMode, 'FDD')
        timeStepLogs = logInfo.DLTimeStepLogs;
        freqAllocations = timeStepLogs(:, 4);
    end
    numOfSlots = size(timeStepLogs, 1) - 1;
    if ~ratType
        numRBG = size(freqAllocations{2}, 2);
        P = ceil(numResourceBlocks / numRBG);
        numRBsPerRBG = P * ones(1, numRBG);
        if mod(numResourceBlocks, P) > 0, numRBsPerRBG(end) = mod(numResourceBlocks, P); end
    end
    avgUEsPerRB = zeros(1, numOfSlots);
    for slotIdx = 1:numOfSlots
        if strcmp(duplexMode, 'TDD')
            slotType = timeStepLogs{slotIdx + 1, 4};
            if ~strcmp(slotType, 'DL'), continue; end
        end
        freqAllocation = freqAllocations{slotIdx + 1};
        if ~ratType
            totalUniqueUEs = sum(arrayfun(@(rbgIdx) nnz(freqAllocation(:, rbgIdx) > 0) * numRBsPerRBG(rbgIdx), 1:length(numRBsPerRBG)));
            avgUEsPerRB(slotIdx) = totalUniqueUEs / numResourceBlocks;
        else
            ueRBUsage = zeros(1, numResourceBlocks);
            for ueIdx = 1:size(freqAllocation, 1)
                startRB = freqAllocation(ueIdx, 1);
                ueRBUsage(startRB + 1:(startRB + freqAllocation(ueIdx, 2))) = ueRBUsage(startRB + 1:(startRB + freqAllocation(ueIdx, 2))) + 1;
            end
            avgUEsPerRB(slotIdx) = mean(ueRBUsage(ueRBUsage > 0));
        end
    end
    avgUEsPerRB = avgUEsPerRB(avgUEsPerRB > 0);
end

function [pairingCounts, pairingDistribution] = calculateUEPairingStats(logInfo, numResourceBlocks, ratType, duplexMode)
    % Calculate UE pairing statistics per RBG
    % Returns:
    %   pairingCounts: Nx2 matrix [numUEs, count] - histogram of how many UEs paired
    %   pairingDistribution: cell array with per-slot pairing info
    
    if strcmp(duplexMode, 'TDD')
        timeStepLogs = logInfo.TimeStepLogs;
        freqAllocations = timeStepLogs(:, 5);
    elseif strcmp(duplexMode, 'FDD')
        timeStepLogs = logInfo.DLTimeStepLogs;
        freqAllocations = timeStepLogs(:, 4);
    end
    
    numOfSlots = size(timeStepLogs, 1) - 1;
    
    % Determine number of RBGs (for RAT0)
    if ~ratType && numOfSlots > 0
        numRBG = size(freqAllocations{2}, 2);
    else
        numRBG = numResourceBlocks; % RAT1: use RBs directly
    end
    
    % Count pairing occurrences: pairingMap(numUEs+1) = count
    % Index 1 = 0 UEs, Index 2 = 1 UE, Index 3 = 2 UEs, etc.
    maxPossibleUEs = 16; % Maximum UEs that can be paired (MU-MIMO limit)
    pairingMap = zeros(1, maxPossibleUEs + 1);
    
    pairingDistribution = cell(numOfSlots, 1);
    
    for slotIdx = 1:numOfSlots
        % Skip non-DL slots in TDD
        if strcmp(duplexMode, 'TDD')
            slotType = timeStepLogs{slotIdx + 1, 4};
            if ~strcmp(slotType, 'DL'), continue; end
        end
        
        freqAllocation = freqAllocations{slotIdx + 1};
        
        if ~ratType
            % RAT0: freqAllocation is [numUEs x numRBGs] binary matrix
            slotPairing = zeros(1, numRBG);
            for rbgIdx = 1:numRBG
                % Count unique UEs on this RBG (non-zero entries in column)
                numUEsOnRBG = nnz(freqAllocation(:, rbgIdx) > 0);
                slotPairing(rbgIdx) = numUEsOnRBG;
                pairingMap(numUEsOnRBG + 1) = pairingMap(numUEsOnRBG + 1) + 1;
            end
        else
            % RAT1: freqAllocation is [numUEs x 2] with [startRB, numRBs]
            % Need to count UEs per RB then aggregate
            uePerRB = zeros(1, numResourceBlocks);
            for ueIdx = 1:size(freqAllocation, 1)
                startRB = freqAllocation(ueIdx, 1);
                numRBs = freqAllocation(ueIdx, 2);
                if numRBs > 0
                    uePerRB(startRB + 1 : startRB + numRBs) = ...
                        uePerRB(startRB + 1 : startRB + numRBs) + 1;
                end
            end
            slotPairing = uePerRB;
            % Count pairing per RB for RAT1
            for rb = 1:numResourceBlocks
                numUEsOnRB = uePerRB(rb);
                pairingMap(numUEsOnRB + 1) = pairingMap(numUEsOnRB + 1) + 1;
            end
        end
        
        pairingDistribution{slotIdx} = slotPairing;
    end
    
    % Convert to Nx2 format [numUEs, count], only include non-zero counts
    validIdx = find(pairingMap > 0);
    pairingCounts = zeros(length(validIdx), 2);
    for i = 1:length(validIdx)
        pairingCounts(i, 1) = validIdx(i) - 1;  % numUEs (0-indexed in map)
        pairingCounts(i, 2) = pairingMap(validIdx(i));
    end
end
