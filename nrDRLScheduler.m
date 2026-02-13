classdef nrDRLScheduler < nrScheduler
    %nrDRLScheduler DRL-based scheduler for 5G NR MU-MIMO downlink scheduling
    %   This class extends nrScheduler to support Deep Reinforcement Learning
    %   based user selection for MU-MIMO scheduling (1LDS architecture).
    %
    %   The scheduler inherits all functionality from nrScheduler including:
    %   - HARQ retransmission handling
    %   - Reserved resources management
    %   - Buffer status management
    %   - Throughput averaging
    %   - MCS selection based on CQI
    %   - PDCCH/PDSCH grant creation
    %
    %   Only the UE selection per RBG (and per MU-MIMO layer) is replaced
    %   by DRL action.
    %
    %   DRL Action Definition (1LDS - One Layer per Decision Stage):
    %   - Each decision stage corresponds to 1 MU-MIMO user-layer
    %   - For each layer l, the agent outputs decisions for all RBGs in one forward pass
    %   - Action matrix A has size [L x NRBG] where L=16 (number of layers)
    %   - A(l,m) ∈ {1..|U|+1}:
    %       * 1..|U|: select the i-th UE from the eligible UEs list
    %       * |U|+1: no allocation for RBG m at layer l
    %
    %   Action Masking:
    %   - Same UE cannot be assigned twice in the same RBG (across layers)
    %   - RBGs occupied by retransmissions cannot be allocated
    %
    %   Usage (Manual Action):
    %       % Create scheduler
    %       scheduler = nrDRLScheduler();
    %       scheduler.EnableDRL = true;
    %       scheduler.NumLayers = 16;
    %
    %       % Each TTI, set action before calling scheduler
    %       actionMatrix = zeros(16, numRBGs);  % [L x NRBG]
    %       % actionMatrix(l,m) = i : assign eligibleUEs(i) to layer l, RBG m
    %       % actionMatrix(l,m) = |U|+1 : no allocation
    %       scheduler.DRLAction = actionMatrix;
    %
    %       % Scheduler will automatically use DRL action
    %       % when EnableDRL=true and ResourceAllocationType=0 (RAT-0)
    %
    %   Usage (Socket with Python DRL Server):
    %       % Create scheduler
    %       scheduler = nrDRLScheduler();
    %       scheduler.EnableDRL = true;
    %       scheduler.NumLayers = 16;
    %
    %       % Connect to Python DRL server
    %       success = scheduler.connectToDRLAgent('127.0.0.1', 5555);
    %       assert(success, 'Cannot connect to Python DRL server');
    %
    %       % Scheduler will automatically:
    %       % 1. Send state to Python each TTI
    %       % 2. Receive action from Python
    %       % 3. Apply action for UE selection
    %
    %       % Python server should handle JSON messages:
    %       % - Receive: {"eligibleUEs":[1,2,3], "numRBGs":17, ...}
    %       % - Send: {"action": [[1,2,1,...], [3,1,4,...], ...]}
    %
    %   Copyright 2024 - Custom DRL Extension

    % ===================================================================
    % DRL MOD START - Properties for DRL integration
    % ===================================================================
    properties (Access = public)
        %DRLAction Action matrix from DRL agent [NumLayers x NumRBGs]
        % Each element A(l,m) specifies:
        %   1..|U|: index of UE in eligibleUEs to allocate at layer l, RBG m
        %   |U|+1:  no allocation (skip this layer-RBG combination)
        % Must be set externally before each TTI scheduling call
        DRLAction = []

        %EnableDRL Flag to enable/disable DRL-based scheduling
        % When true: use DRL action for UE selection
        % When false: use default nrScheduler behavior
        EnableDRL = true

        %NumLayers Number of MU-MIMO spatial layers (default: 16)
        NumLayers = 16

        %SchedulingStrategy Strategy identifier
        % "DRL" - use DRL action
        % "Default" - use nrScheduler default (round-robin/PF/BestCQI)
        SchedulingStrategy = "DRL"

        %MaxUsersPerRBG Maximum number of UEs that can be co-scheduled on same RBG
        % This is typically limited by MU-MIMO capability
        MaxUsersPerRBG = 16
        
        %EnableI1Constraint Enable PMI i1 matching constraint for MU-MIMO pairing
        % When true: UEs co-scheduled on same RBG must have same i1 (wideband PMI)
        % This ensures good beam orthogonality and reduces inter-user interference
        % When false: no i1 constraint (original behavior)
        EnableI1Constraint = true
        
        %EnableOrthogonalityConstraint Enable precoder orthogonality check for MU-MIMO
        % When true: Check that paired UEs have orthogonal precoders (|W1'*W2| < threshold)
        % This is essential for CSI-RS based MU-MIMO to reduce inter-user interference
        EnableOrthogonalityConstraint = true
        
        %SemiOrthogonalityFactor Threshold for precoder orthogonality (0-1)
        % Two precoders are considered orthogonal if |W1'*W2|/max < (1 - SemiOrthogonalityFactor)
        % Higher value = stricter orthogonality requirement = fewer pairings but less interference
        % Typical values: 0.5-0.8
        SemiOrthogonalityFactor = 0.7
        
        %MU_MCSBackoff MCS reduction for MU-MIMO transmissions
        % Reduces MCS when multiple UEs share RBG to account for residual interference
        % Value is number of MCS levels to reduce per co-scheduled UE (beyond first)
        MU_MCSBackoff = 2

        %DRLDebug Enable debug output for DRL scheduling
        DRLDebug = false

        %LastDRLAllocation Store the last DRL allocation for debugging/logging
        LastDRLAllocation = struct('allottedUEs', [], 'freqAllocation', [], ...
            'layerAssignment', [], 'eligibleUEs', [])

        % ---------------------------------------------------------------
        % Socket Communication Properties
        % ---------------------------------------------------------------
        %DRL_IP IP address of Python DRL server
        DRL_IP = "127.0.0.1"

        %DRL_Port TCP port of Python DRL server
        DRL_Port = 5555

        %DRL_Socket TCP client socket object
        DRL_Socket = []

        %DRL_IsConnected Flag indicating socket connection status
        DRL_IsConnected = false

        %DRL_RxBuf Receive buffer for incoming data
        DRL_RxBuf uint8 = uint8([])

        %DRL_Terminator Line terminator character (newline)
        DRL_Terminator = uint8(10)

        %DRL_TimeoutSec Socket read timeout in seconds
        DRL_TimeoutSec = 5

        %UseSocket Flag to enable socket-based action retrieval
        % When true: automatically fetch action from Python server
        % When false: use manually set DRLAction property
        UseSocket = false

        %MaxUEs Maximum number of UEs (for feature matrix sizing)
        MaxUEs = 16

        %SubbandSize RBs per feature subband (for CQI features)
        SubbandSize = 16
    end
    % ===================================================================
    % DRL MOD END - Properties
    % ===================================================================

    methods (Access = public)
        function obj = nrDRLScheduler(varargin)
            %nrDRLScheduler Constructor
            % Calls parent constructor and initializes DRL-specific properties
        end

        function setDRLAction(obj, actionMatrix)
            %setDRLAction Set the DRL action matrix for current TTI
            %   obj.setDRLAction(actionMatrix) sets the action matrix A
            %   where A is [NumLayers x NumRBGs]
            %
            %   actionMatrix(l,m) ∈ {1..|U|+1}:
            %       1..|U|: allocate eligible UE at index i to layer l, RBG m
            %       |U|+1:  no allocation

            obj.DRLAction = actionMatrix;
            if obj.DRLDebug
                fprintf('[nrDRLScheduler] DRLAction set: size=[%d x %d]\n', ...
                    size(actionMatrix,1), size(actionMatrix,2));
            end
        end

        function clearDRLAction(obj)
            %clearDRLAction Clear the DRL action matrix
            obj.DRLAction = [];
        end

        % ---------------------------------------------------------------
        % Socket Communication Methods
        % ---------------------------------------------------------------
        function success = connectToDRLAgent(obj, ip, port)
            %connectToDRLAgent Connect to Python DRL server via TCP socket
            %   success = connectToDRLAgent(obj) connects using default IP/port
            %   success = connectToDRLAgent(obj, ip, port) connects to specified address
            %
            %   Returns true if connection successful, false otherwise

            if nargin >= 2 && ~isempty(ip)
                obj.DRL_IP = ip;
            end
            if nargin >= 3 && ~isempty(port)
                obj.DRL_Port = port;
            end

            success = false;

            % Clean up existing socket
            if ~isempty(obj.DRL_Socket)
                try
                    delete(obj.DRL_Socket);
                catch
                end
                obj.DRL_Socket = [];
            end

            try
                obj.DRL_Socket = tcpclient(obj.DRL_IP, obj.DRL_Port, ...
                    'Timeout', 60, 'ConnectTimeout', 60);
                obj.DRL_Socket.InputBufferSize  = 1048576;
                obj.DRL_Socket.OutputBufferSize = 1048576;
                obj.DRL_IsConnected = true;
                obj.UseSocket = true;
                fprintf('[nrDRLScheduler] Connected to Python DRL at %s:%d\n', ...
                    obj.DRL_IP, obj.DRL_Port);
                success = true;
            catch ME
                fprintf('[nrDRLScheduler] Connection failed: %s\n', ME.message);
                obj.DRL_IsConnected = false;
                obj.UseSocket = false;
            end
        end

        function disconnectFromDRLAgent(obj)
            %disconnectFromDRLAgent Disconnect from Python DRL server

            if ~isempty(obj.DRL_Socket)
                try
                    delete(obj.DRL_Socket);
                catch
                end
                obj.DRL_Socket = [];
            end
            obj.DRL_IsConnected = false;
            fprintf('[nrDRLScheduler] Disconnected from Python DRL\n');
        end

        function success = sendStateToPython(obj, stateStruct)
            %sendStateToPython Send scheduler state to Python DRL agent
            %   success = sendStateToPython(obj, stateStruct)
            %
            %   stateStruct should contain:
            %     - eligibleUEs: RNTI list
            %     - numEligibleUEs: count
            %     - numRBGs: number of RBGs
            %     - freqOccupancyBitmap: retransmission occupancy
            %     - channelQuality: CQI per UE
            %     - bufferStatus: buffer per UE
            %     - Any other features for DRL

            success = false;
            if ~obj.DRL_IsConnected || isempty(obj.DRL_Socket)
                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Not connected, cannot send state\n');
                end
                return;
            end

            try
                % Convert struct to JSON string
                jsonStr = jsonencode(stateStruct);
                % Send with newline terminator
                write(obj.DRL_Socket, uint8([jsonStr, newline]));
                success = true;
                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Sent state: %d bytes\n', length(jsonStr));
                end
            catch ME
                fprintf('[nrDRLScheduler] Send error: %s\n', ME.message);
                obj.DRL_IsConnected = false;
            end
        end

        function actionMatrix = receiveActionFromPython(obj, numLayers, numRBGs, numEligibleUEs)
            %receiveActionFromPython Receive DRL action from Python server
            %   actionMatrix = receiveActionFromPython(obj, numLayers, numRBGs, numEligibleUEs)
            %
            %   Returns action matrix [numLayers x numRBGs]
            %   On error, returns default "no allocation" matrix

            % Default: no allocation (all = numEligibleUEs + 1)
            noAllocAction = numEligibleUEs + 1;
            actionMatrix = noAllocAction * ones(numLayers, numRBGs);

            if ~obj.DRL_IsConnected || isempty(obj.DRL_Socket)
                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Not connected, using default action\n');
                end
                return;
            end

            try
                % Read response line from Python
                jsonStr = obj.readLineFromSocket();
                if isempty(jsonStr)
                    if obj.DRLDebug
                        fprintf('[nrDRLScheduler] Empty response from Python\n');
                    end
                    return;
                end

                % Parse JSON response
                response = jsondecode(jsonStr);

                % Extract action matrix
                if isfield(response, 'action')
                    rawAction = response.action;
                    [aRows, aCols] = size(rawAction);

                    % Validate and adjust dimensions
                    actualLayers = min(aRows, numLayers);
                    actualRBGs = min(aCols, numRBGs);

                    % Copy valid portion
                    actionMatrix(1:actualLayers, 1:actualRBGs) = rawAction(1:actualLayers, 1:actualRBGs);

                    % Clamp values to valid range [1, numEligibleUEs+1]
                    actionMatrix = max(1, min(noAllocAction, actionMatrix));

                    if obj.DRLDebug
                        fprintf('[nrDRLScheduler] Received action: [%d x %d]\n', aRows, aCols);
                    end
                else
                    if obj.DRLDebug
                        fprintf('[nrDRLScheduler] Response missing "action" field\n');
                    end
                end
            catch ME
                fprintf('[nrDRLScheduler] Receive error: %s\n', ME.message);
            end
        end

        function line = readLineFromSocket(obj)
            %readLineFromSocket Read a complete line (until newline) from socket
            %   Helper method for socket communication

            line = '';
            startTime = tic;

            while toc(startTime) < obj.DRL_TimeoutSec
                if obj.DRL_Socket.NumBytesAvailable > 0
                    newData = read(obj.DRL_Socket, obj.DRL_Socket.NumBytesAvailable, 'uint8');
                    obj.DRL_RxBuf = [obj.DRL_RxBuf, newData];
                end

                % Check for terminator
                termIdx = find(obj.DRL_RxBuf == obj.DRL_Terminator, 1);
                if ~isempty(termIdx)
                    line = char(obj.DRL_RxBuf(1:termIdx-1));
                    obj.DRL_RxBuf = obj.DRL_RxBuf(termIdx+1:end);
                    return;
                end

                pause(0.001); % Small delay to avoid busy-waiting
            end

            if obj.DRLDebug
                fprintf('[nrDRLScheduler] Socket read timeout\n');
            end
        end

        function actionMatrix = requestActionFromPython(obj, schedulerInput)
            %requestActionFromPython Full round-trip using TTI_OBS/TTI_ALLOC protocol
            %   actionMatrix = requestActionFromPython(obj, schedulerInput)
            %
            %   Protocol (compatible with train_drl_with_matlab.py):
            %   1. Send TTI_OBS message with features
            %   2. Receive TTI_ALLOC message with allocation matrix
            %   3. Convert allocation to DRLAction format
            %
            %   Observation features sent (matching Python _build_obs):
            %   - avg_throughput: [MaxUEs] - normalized avg throughput per UE
            %   - rank: [MaxUEs] - rank per UE
            %   - buffer: [MaxUEs] - buffer status per UE  
            %   - wideband_cqi: [MaxUEs] - wideband CQI per UE
            %   - subband_cqi: [MaxUEs x numRBGs] - subband CQI per UE per RBG
            %   - cross_corr: [MaxUEs x MaxUEs x numRBGs] - precoder correlation matrix

            eligibleUEs = schedulerInput.eligibleUEs;
            numEligibleUEs = numel(eligibleUEs);
            numRBGs = numel(schedulerInput.freqOccupancyBitmap);
            numRBs = obj.CellConfig(1).NumResourceBlocks;

            % Default: no allocation
            noAllocAction = numEligibleUEs + 1;
            actionMatrix = noAllocAction * ones(obj.NumLayers, numRBGs);

            if ~obj.DRL_IsConnected || isempty(obj.DRL_Socket)
                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Not connected, using default action\n');
                end
                return;
            end

            % ---------------------------------------------------------------
            % Build observation features matching Python _build_obs format
            % ---------------------------------------------------------------
            
            % 1. Average throughput per UE [MaxUEs]
            avg_throughput = zeros(obj.MaxUEs, 1);
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                if rnti <= obj.MaxUEs
                    ueCtx = obj.UEContext(rnti);
                    % Get average throughput from OLLA or throughput history
                    if isprop(ueCtx, 'AverageThroughputDL')
                        avg_throughput(rnti) = ueCtx.AverageThroughputDL;
                    else
                        avg_throughput(rnti) = 0;
                    end
                end
            end
            
            % 2. Rank per UE [MaxUEs]
            rank_ue = zeros(obj.MaxUEs, 1);
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                if rnti <= obj.MaxUEs && i <= numel(schedulerInput.selectedRank)
                    rank_ue(rnti) = schedulerInput.selectedRank(i);
                end
            end
            
            % 3. Buffer status per UE [MaxUEs]
            buffer_status = zeros(obj.MaxUEs, 1);
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                if rnti <= obj.MaxUEs
                    ueCtx = obj.UEContext(rnti);
                    buffer_status(rnti) = ueCtx.BufferStatusDL;
                end
            end
            
            % 4. Wideband CQI per UE [MaxUEs]
            wideband_cqi = zeros(obj.MaxUEs, 1);
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                if rnti <= obj.MaxUEs
                    wideband_cqi(rnti) = mean(schedulerInput.channelQuality(i, :));
                end
            end
            
            % 5. Subband CQI per UE per RBG [MaxUEs x numRBGs]
            subband_cqi = zeros(obj.MaxUEs, numRBGs);
            rbgSize = ceil(numRBs / numRBGs);
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                if rnti <= obj.MaxUEs
                    for rbg = 1:numRBGs
                        rbStart = (rbg-1)*rbgSize + 1;
                        rbEnd = min(rbg*rbgSize, numRBs);
                        subband_cqi(rnti, rbg) = mean(schedulerInput.channelQuality(i, rbStart:rbEnd));
                    end
                end
            end
            
            % 6. Precoder cross-correlation matrix [MaxUEs x MaxUEs x numRBGs]
            %    cross_corr(u1, u2, rbg) = |W_u1' * W_u2| (normalized)
            cross_corr = zeros(obj.MaxUEs, obj.MaxUEs, numRBGs);
            W = schedulerInput.W;  % Cell array of precoders
            
            for i = 1:numEligibleUEs
                rnti_i = eligibleUEs(i);
                if rnti_i > obj.MaxUEs, continue; end
                W_i = W{i};
                if isempty(W_i) || isreal(W_i), continue; end
                
                for j = i:numEligibleUEs
                    rnti_j = eligibleUEs(j);
                    if rnti_j > obj.MaxUEs, continue; end
                    W_j = W{j};
                    if isempty(W_j) || isreal(W_j), continue; end
                    
                    % Compute correlation per RBG
                    for rbg = 1:numRBGs
                        try
                            if ndims(W_i) >= 3 && ndims(W_j) >= 3
                                % Subband precoders: W is 3D
                                numSB = min(size(W_i, 3), size(W_j, 3));
                                % Map RBG to subband
                                sbIdx = min(rbg, numSB);
                                w1 = W_i(:,:,sbIdx);
                                w2 = W_j(:,:,sbIdx);
                            else
                                % Wideband precoders
                                w1 = W_i;
                                w2 = W_j;
                            end
                            
                            % Flatten and compute normalized correlation
                            w1_flat = w1(:) / (norm(w1(:)) + 1e-10);
                            w2_flat = w2(:) / (norm(w2(:)) + 1e-10);
                            corr_val = abs(w1_flat' * w2_flat);
                            
                            % Symmetric matrix
                            cross_corr(rnti_i, rnti_j, rbg) = corr_val;
                            cross_corr(rnti_j, rnti_i, rbg) = corr_val;
                        catch
                            % Skip if computation fails
                        end
                    end
                end
            end

            % Build TTI_OBS payload with new observation format
            payload = struct();
            payload.type = "TTI_OBS";
            payload.frame = double(obj.CurrFrame);
            payload.slot = double(obj.CurrSlot);
            payload.max_ues = obj.MaxUEs;
            payload.max_layers = obj.NumLayers;
            payload.num_rbg = numRBGs;
            payload.num_rbs = numRBs;
            payload.eligible_ues = eligibleUEs;
            
            % Observation features matching Python _build_obs format
            payload.avg_throughput = avg_throughput;       % [MaxUEs]
            payload.rank = rank_ue;                        % [MaxUEs]  
            payload.buffer = buffer_status;                % [MaxUEs]
            payload.wideband_cqi = wideband_cqi;           % [MaxUEs]
            payload.subband_cqi = subband_cqi;             % [MaxUEs x numRBGs]
            payload.cross_corr = cross_corr;               % [MaxUEs x MaxUEs x numRBGs]

            try
                % Send TTI_OBS
                jsonStr = jsonencode(payload);
                write(obj.DRL_Socket, uint8([jsonStr, newline]));

                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Sent TTI_OBS: frame=%d slot=%d\n', ...
                        payload.frame, payload.slot);
                end

                % Receive TTI_ALLOC
                respStr = obj.readLineFromSocket();
                if isempty(respStr)
                    fprintf('[nrDRLScheduler] Empty response from Python\n');
                    return;
                end

                resp = jsondecode(respStr);

                if ~isfield(resp, 'type') || ~strcmp(resp.type, 'TTI_ALLOC')
                    fprintf('[nrDRLScheduler] Invalid response type\n');
                    return;
                end

                % allocation is [numRBGs x MaxNumLayers] with RNTI values (0 = no alloc)
                allocation = resp.allocation;

                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Received TTI_ALLOC: size=[%d x %d]\n', ...
                        size(allocation, 1), size(allocation, 2));
                end

                % Convert allocation (RNTI) to actionMatrix (eligibleUE index)
                % allocation[rbg][layer] = RNTI or 0
                % actionMatrix[layer][rbg] = index in eligibleUEs or numEligibleUEs+1
                for rbg = 1:numRBGs
                    for layer = 1:obj.NumLayers
                        if rbg <= size(allocation, 1) && layer <= size(allocation, 2)
                            rnti = allocation(rbg, layer);
                            if rnti > 0
                                ueIdx = find(eligibleUEs == rnti, 1);
                                if ~isempty(ueIdx)
                                    actionMatrix(layer, rbg) = ueIdx;
                                end
                            end
                        end
                    end
                end

            catch ME
                fprintf('[nrDRLScheduler] Communication error: %s\n', ME.message);
            end
        end
    end

    methods (Access = protected)
        function dlAssignments = scheduleNewTransmissionsDL(obj, timeFrequencyResource, schedulingInfo)
            %scheduleNewTransmissionsDL Assign resources for new DL transmissions
            %   Overrides the parent method to inject DRL-based UE selection
            %   while keeping all other scheduler logic intact.
            %
            %   When EnableDRL=true and ResourceAllocationType=0 (RAT-0):
            %       Uses runSchedulingStrategyDRL_RAT0 for UE selection
            %   Otherwise:
            %       Calls parent scheduleNewTransmissionsDL method

            % ===================================================================
            % DRL MOD START - Check DRL mode and RAT-0
            % ===================================================================
            schedulerConfig = obj.SchedulerConfig;

            % Check if we should use DRL scheduling
            % Note: if UseSocket=true, DRLAction will be fetched automatically
            useDRL = obj.EnableDRL && ...
                     (obj.UseSocket || ~isempty(obj.DRLAction)) && ...
                     (obj.SchedulingStrategy == "DRL") && ...
                     (schedulerConfig.ResourceAllocationType == 0);  % RAT-0 only

            if useDRL
                % Use DRL-based scheduling
                dlAssignments = obj.scheduleNewTransmissionsDL_DRL(timeFrequencyResource, schedulingInfo);
            else
                % Use default nrScheduler behavior
                dlAssignments = scheduleNewTransmissionsDL@nrScheduler(obj, timeFrequencyResource, schedulingInfo);
            end
            % ===================================================================
            % DRL MOD END - Check DRL mode
            % ===================================================================
        end

        % ===================================================================
        % DRL MOD START - DRL-based scheduling method
        % ===================================================================
        function dlAssignments = scheduleNewTransmissionsDL_DRL(obj, timeFrequencyResource, schedulingInfo)
            %scheduleNewTransmissionsDL_DRL DRL-based DL scheduling implementation
            %   This method follows the exact same flow as the original
            %   scheduleNewTransmissionsDL but replaces the UE selection
            %   logic with DRL action interpretation.

            % Read time-frequency resources (same as original)
            scheduledSlot = timeFrequencyResource.NSlot; %#ok<NASGU>
            startSym = timeFrequencyResource.SymbolAllocation(1); %#ok<NASGU>
            numSym = timeFrequencyResource.SymbolAllocation(2);
            updatedFrequencyStatus = timeFrequencyResource.FrequencyResource;

            % Read scheduling info structure (same as original)
            eligibleUEs = schedulingInfo.EligibleUEs;
            numNewTxs = min(size(eligibleUEs,2), schedulingInfo.MaxNumUsersTTI);

            % Early return if no eligible UEs
            if isempty(eligibleUEs)
                dlAssignments = struct([]);
                return;
            end

            % Pre-allocate DL grants (same as original)
            dlAssignments = obj.DLGrantArrayStruct(1:numNewTxs);

            cellConfig = obj.CellConfig;
            schedulerConfig = obj.SchedulerConfig;
            ueContext = obj.UEContext;

            % ---------------------------------------------------------------
            % Select rank and precoding matrix for the eligible UEs
            % (Reuse logic from parent class)
            % ---------------------------------------------------------------
            numEligibleUEs = size(eligibleUEs,2);
            W = cell(numEligibleUEs, 1);
            H = cell(numEligibleUEs, 1);
            SINR = cell(numEligibleUEs, 1);
            selectedRanks = zeros(numEligibleUEs, 1);  % Renamed from 'rank' to avoid conflict with built-in
            rbRequirement = zeros(obj.NumUEs, 1);
            channelQuality = zeros(obj.NumUEs, cellConfig.NumResourceBlocks);
            cqiSizeArray = ones(cellConfig.NumResourceBlocks, 1);
            nVar = 0;

            % ----------------------------------------------------------
            % DRL MOD: Pre-allocate i1 storage for PMI constraint
            % i1 is wideband PMI component - UEs with same i1 have
            % similar beam direction and can be paired for MU-MIMO
            % ----------------------------------------------------------
            ueI1Values = cell(numEligibleUEs, 1);  % Store PMI i1 for each eligible UE
            
            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                eligibleUEContext = ueContext(rnti);
                carrierContext = eligibleUEContext.ComponentCarrier(1);
                csiMeasurement = carrierContext.CSIMeasurementDL;
                csiMeasurementCQI = max(csiMeasurement.CSIRS.CQI(:)) * cqiSizeArray(1);
                channelQuality(rnti, :) = csiMeasurementCQI;

                if isempty(carrierContext.CSIRSConfiguration)
                    numCSIRSPorts = obj.CellConfig(1).NumTransmitAntennas;
                else
                    numCSIRSPorts = carrierContext.CSIRSConfiguration.NumCSIRSPorts;
                end

                [selectedRanks(i), W{i}] = selectRankAndPrecodingMatrixDL(obj, rnti, csiMeasurement, numCSIRSPorts);

                % DRL MOD: Extract PMI i1 for pairing constraint
                % i1(1:2) contains [i11, i12] - the wideband beam indices
                if isfield(csiMeasurement, 'CSIRS') && ...
                   isfield(csiMeasurement.CSIRS, 'PMISet') && ...
                   isfield(csiMeasurement.CSIRS.PMISet, 'i1') && ...
                   numel(csiMeasurement.CSIRS.PMISet.i1) >= 2
                    ueI1Values{i} = csiMeasurement.CSIRS.PMISet.i1(1:2);
                else
                    % If no i1 available, use empty (will skip i1 check)
                    ueI1Values{i} = [];
                end

                % For SRS-based DL MU-MIMO CSI measurements
                if ~isempty(schedulerConfig.MUMIMOConfigDL) && ~isempty(csiMeasurement.SRS)
                    H{i} = csiMeasurement.SRS.H;
                    nVar = csiMeasurement.SRS.nVar;
                    SINR{i} = csiMeasurement.SRS.sinr;
                end

                [~, rbRequirement(rnti)] = calculateRBRequirement(obj, rnti, obj.DLType, numSym, selectedRanks(i));
            end

            % ---------------------------------------------------------------
            % Create scheduler input structure (same format as original)
            % ---------------------------------------------------------------
            schedulerInput = obj.SchedulerInputStruct;
            schedulerInput.linkDir = obj.DLType;
            schedulerInput.eligibleUEs = eligibleUEs;
            schedulerInput.selectedRank = selectedRanks;
            schedulerInput.bufferStatus = [ueContext(eligibleUEs).BufferStatusDL];
            schedulerInput.lastSelectedUE = obj.LastSelectedUEDL;
            schedulerInput.channelQuality = channelQuality(eligibleUEs, :);
            schedulerInput.freqOccupancyBitmap = updatedFrequencyStatus;
            schedulerInput.rbAllocationLimit = obj.RBAllocationLimitDL;
            schedulerInput.rbRequirement = rbRequirement(eligibleUEs);
            schedulerInput.maxNumUsersTTI = schedulingInfo.MaxNumUsersTTI;
            schedulerInput.numSym = numSym;
            schedulerInput.W = W;
            schedulerInput.ueI1Values = ueI1Values;  % DRL MOD: Store i1 for pairing constraint

            % For SRS-based MU-MIMO
            csiMeasurementSignalDLType = obj.SchedulerConfig.CSIMeasurementSignalDLType;
            if ~isempty(schedulerConfig.MUMIMOConfigDL) && csiMeasurementSignalDLType
                isSRSApplicable = any(arrayfun(@(x) ~isempty(obj.UEContext(x).CSIMeasurementDL.SRS), eligibleUEs));
                if isSRSApplicable
                    schedulerInput.channelMatrix = H;
                    schedulerInput.nVar = nVar;
                    schedulerInput.SINRs = SINR;
                end
            end

            % ---------------------------------------------------------------
            % DRL-based UE selection (replaces runSchedulingStrategyRAT0)
            % ---------------------------------------------------------------
            [allottedUEs, freqAllocation, mcsIndex, W] = runSchedulingStrategyDRL_RAT0(obj, schedulerInput);

            % ---------------------------------------------------------------
            % Create DL assignments (same as original)
            % ---------------------------------------------------------------
            numAllottedUEs = numel(allottedUEs);
            for index = 1:numAllottedUEs
                gNBCarrierIndex = 1;
                dlAssignments(index).GNBCarrierIndex = gNBCarrierIndex;
                selectedUE = allottedUEs(index);

                % Find UE index in eligible UEs set
                selectedUEIdx = find(eligibleUEs == selectedUE, 1);

                % MCS offset value
                carrierContext = ueContext(selectedUE).ComponentCarrier(gNBCarrierIndex);
                mcsOffset = fix(carrierContext.MCSOffset(schedulerInput.linkDir+1));

                % Fill DL assignment properties
                dlAssignments(index).RNTI = selectedUE;
                dlAssignments(index).FrequencyAllocation = freqAllocation(index, :);
                dlAssignments(index).MCSIndex = min(max(mcsIndex(index) - mcsOffset, 0), 27);
                dlAssignments(index).W = W{selectedUEIdx};

                % Mark frequency resources as assigned
                updatedFrequencyStatus = updatedFrequencyStatus | freqAllocation(index,:);
            end

            % Remove invalid trailing entries - ensure column vector for concatenation
            if numAllottedUEs > 0
                dlAssignments = dlAssignments(1:numAllottedUEs);
                % Ensure column vector (required by parent nrScheduler)
                dlAssignments = dlAssignments(:);
            else
                % Empty case: return empty struct array compatible with parent
                dlAssignments = struct([]);
            end

            % Update last selected UE
            if numAllottedUEs > 0
                obj.LastSelectedUEDL = allottedUEs(end);
            end
        end
        % ===================================================================
        % DRL MOD END - DRL-based scheduling method
        % ===================================================================

        % ===================================================================
        % DRL MOD START - DRL scheduling strategy for RAT-0
        % ===================================================================
        function [allottedUEs, freqAllocation, mcsIndex, W] = runSchedulingStrategyDRL_RAT0(obj, schedulerInput)
            %runSchedulingStrategyDRL_RAT0 DRL-based UE selection for RAT-0
            %   This function implements DRL action interpretation for MU-MIMO
            %   scheduling with Resource Allocation Type 0 (RBG-based bitmap).
            %
            %   [ALLOTTEDUES, FREQALLOCATION, MCSINDEX, W] =
            %   runSchedulingStrategyDRL_RAT0(OBJ, SCHEDULERINPUT) returns:
            %
            %   ALLOTTEDUES - Vector of RNTIs actually allocated (unique)
            %   FREQALLOCATION - Matrix [numAllottedUEs x numRBGs] with RBG bitmap per UE
            %   MCSINDEX - Vector of MCS indices for each allotted UE
            %   W - Cell array of precoding matrices for each allotted UE
            %
            %   DRL Action Interpretation:
            %   - obj.DRLAction is [NumLayers x NumRBGs] matrix
            %   - DRLAction(l,m) ∈ {1..|U|+1}:
            %       * 1..|U|: allocate eligibleUEs(i) to layer l, RBG m
            %       * |U|+1:  no allocation (skip)
            %   - Action masking enforces:
            %       * No duplicate UE allocation on same RBG across layers
            %       * No allocation on RBGs occupied by retransmissions

            eligibleUEs = schedulerInput.eligibleUEs;
            numEligibleUEs = numel(eligibleUEs);
            rbgOccupancyBitmap = schedulerInput.freqOccupancyBitmap;
            numRBGs = numel(rbgOccupancyBitmap);

            % ---------------------------------------------------------------
            % Get DRL action: from socket or from property
            % ---------------------------------------------------------------
            if obj.UseSocket && obj.DRL_IsConnected
                % Request action from Python DRL server
                drlAction = obj.requestActionFromPython(schedulerInput);
                obj.DRLAction = drlAction; % Store for debugging
            else
                % Use manually set action
                drlAction = obj.DRLAction;
            end

            % Handle empty or invalid action
            if isempty(drlAction)
                if obj.DRLDebug
                    fprintf('[nrDRLScheduler] Empty DRLAction, using no-allocation default\n');
                end
                drlAction = (numEligibleUEs + 1) * ones(obj.NumLayers, numRBGs);
            end

            % Validate DRL action dimensions
            [actionNumLayers, actionNumRBGs] = size(drlAction);
            numLayers = min(actionNumLayers, obj.NumLayers);

            % Handle dimension mismatch
            if actionNumRBGs ~= numRBGs
                if obj.DRLDebug
                    warning('[nrDRLScheduler] DRLAction RBG mismatch: action=%d, expected=%d', ...
                        actionNumRBGs, numRBGs);
                end
                % Pad or truncate action matrix
                if actionNumRBGs < numRBGs
                    % Pad with "no allocation" action
                    drlAction = [drlAction, (numEligibleUEs+1)*ones(actionNumLayers, numRBGs-actionNumRBGs)];
                else
                    % Truncate
                    drlAction = drlAction(:, 1:numRBGs);
                end
            end

            % Initialize output arrays
            ueRBGAllocation = zeros(obj.NumUEs, numRBGs);  % Full allocation matrix
            ueLayerCount = zeros(obj.NumUEs, numRBGs);     % Track layers per UE per RBG

            % DRL MOD: Get i1 values for i1 matching constraint
            ueI1Values = schedulerInput.ueI1Values;
            
            % DRL MOD: Get precoders for orthogonality check
            W = schedulerInput.W;  % Cell array of precoders per eligible UE
            
            % DRL MOD: Get MaxNumUsersPerTTI constraint
            maxUsersPerTTI = obj.SchedulerConfig.MaxNumUsersPerTTI;
            totalUniqueUEsAllocated = [];  % Track unique UEs across all RBGs
            
            % Helper function for ternary operation
            ternary = @(cond, trueVal, falseVal) subsref({falseVal, trueVal}, struct('type', '{}', 'subs', {{cond + 1}}));
            
            % Process each RBG
            for rbg = 1:numRBGs
                % Skip if RBG is occupied by retransmission
                if rbgOccupancyBitmap(rbg)
                    continue;
                end

                % Track which UEs are already allocated on this RBG (action masking)
                allocatedUEsThisRBG = [];
                allocatedUEIndicesThisRBG = [];  % DRL MOD: Track indices for i1 lookup

                % Process each layer
                for layer = 1:numLayers
                    actionIndex = drlAction(layer, rbg);

                    % Validate action index
                    if actionIndex <= 0 || actionIndex > numEligibleUEs + 1
                        % Invalid action: treat as no allocation
                        continue;
                    end

                    % Action = numEligibleUEs + 1 means "no allocation"
                    if actionIndex == numEligibleUEs + 1
                        continue;
                    end

                    % Get selected UE RNTI
                    selectedUE = eligibleUEs(actionIndex);

                    % Action Masking: Check if UE already allocated on this RBG
                    if ismember(selectedUE, allocatedUEsThisRBG)
                        % Skip duplicate allocation (action masked)
                        if obj.DRLDebug
                            fprintf('[DRL] Layer %d, RBG %d: UE %d already allocated, skipping\n', ...
                                layer, rbg, selectedUE);
                        end
                        continue;
                    end

                    % Check MaxUsersPerRBG constraint
                    if numel(allocatedUEsThisRBG) >= obj.MaxUsersPerRBG
                        if obj.DRLDebug
                            fprintf('[DRL] Layer %d, RBG %d: MaxUsersPerRBG reached, skipping\n', ...
                                layer, rbg);
                        end
                        continue;
                    end
                    
                    % Check MaxNumUsersPerTTI constraint (total unique UEs in this TTI)
                    if ~ismember(selectedUE, totalUniqueUEsAllocated) && ...
                            numel(totalUniqueUEsAllocated) >= maxUsersPerTTI
                        if obj.DRLDebug
                            fprintf('[DRL] Layer %d, RBG %d: MaxNumUsersPerTTI reached (%d), skipping UE %d\n', ...
                                layer, rbg, maxUsersPerTTI, selectedUE);
                        end
                        continue;
                    end
                    
                    % ----------------------------------------------------------
                    % DRL MOD: PMI i1 Matching Constraint
                    % UEs co-scheduled on same RBG must have same i1 (wideband PMI)
                    % This ensures good beam orthogonality for MU-MIMO pairing
                    % ----------------------------------------------------------
                    if obj.EnableI1Constraint && ~isempty(allocatedUEIndicesThisRBG)
                        currentUEI1 = ueI1Values{actionIndex};
                        if ~isempty(currentUEI1)
                            % Check i1 matching with all already allocated UEs on this RBG
                            i1Matched = true;
                            for allocIdx = 1:numel(allocatedUEIndicesThisRBG)
                                existingUEIdx = allocatedUEIndicesThisRBG(allocIdx);
                                existingUEI1 = ueI1Values{existingUEIdx};
                                if ~isempty(existingUEI1) && ~all(currentUEI1 == existingUEI1)
                                    i1Matched = false;
                                    break;
                                end
                            end
                            if ~i1Matched
                                if obj.DRLDebug
                                    fprintf('[DRL] Layer %d, RBG %d: UE %d i1 mismatch, skipping\n', ...
                                        layer, rbg, selectedUE);
                                end
                                continue;
                            end
                        end
                    end
                    
                    % ----------------------------------------------------------
                    % DRL MOD: Precoder Orthogonality Constraint (CSI-RS based)
                    % Check that paired UEs have semi-orthogonal precoders
                    % |W_new' * W_existing| / max < (1 - SemiOrthogonalityFactor)
                    % ----------------------------------------------------------
                    if obj.EnableOrthogonalityConstraint && ~isempty(allocatedUEIndicesThisRBG)
                        W_new = W{actionIndex};
                        if ~isempty(W_new) && ~isreal(W_new)
                            isOrthogonal = true;
                            for allocIdx = 1:numel(allocatedUEIndicesThisRBG)
                                existingUEIdx = allocatedUEIndicesThisRBG(allocIdx);
                                W_existing = W{existingUEIdx};
                                if ~isempty(W_existing) && ~isreal(W_existing)
                                    % Compute correlation between precoders
                                    % Handle both 2D (wideband) and 3D (subband) precoders
                                    % W can be [numPorts x numLayers] or [numLayers x numPorts x numSubbands]
                                    try
                                        if ismatrix(W_new) && ismatrix(W_existing)
                                            % Wideband: flatten to column vectors and compute correlation
                                            w1 = W_new(:);
                                            w2 = W_existing(:);
                                            % Normalize and compute inner product
                                            w1 = w1 / (norm(w1) + 1e-10);
                                            w2 = w2 / (norm(w2) + 1e-10);
                                            maxCorr = abs(w1' * w2);
                                        else
                                            % Subband: W is 3D, take max correlation across subbands
                                            numSB = min(size(W_new, 3), size(W_existing, 3));
                                            maxCorr = 0;
                                            for sb = 1:numSB
                                                w1 = W_new(:,:,sb);
                                                w2 = W_existing(:,:,sb);
                                                w1 = w1(:) / (norm(w1(:)) + 1e-10);
                                                w2 = w2(:) / (norm(w2(:)) + 1e-10);
                                                corrSB = abs(w1' * w2);
                                                maxCorr = max(maxCorr, corrSB);
                                            end
                                        end
                                    catch
                                        % If computation fails, assume not orthogonal (conservative)
                                        maxCorr = 1;
                                    end
                                    
                                    % Check orthogonality threshold
                                    % maxCorr is already normalized to [0,1]
                                    if maxCorr > (1 - obj.SemiOrthogonalityFactor)
                                        isOrthogonal = false;
                                        if obj.DRLDebug
                                            fprintf('[DRL] Layer %d, RBG %d: UE %d not orthogonal to UE %d (corr=%.3f), skipping\n', ...
                                                layer, rbg, selectedUE, allocatedUEsThisRBG(allocIdx), maxCorr);
                                        end
                                        break;
                                    end
                                end
                            end
                            if ~isOrthogonal
                                continue;
                            end
                        end
                    end

                    % Allocate RBG to UE
                    ueRBGAllocation(selectedUE, rbg) = 1;
                    ueLayerCount(selectedUE, rbg) = ueLayerCount(selectedUE, rbg) + 1;
                    allocatedUEsThisRBG = [allocatedUEsThisRBG, selectedUE]; %#ok<AGROW>
                    allocatedUEIndicesThisRBG = [allocatedUEIndicesThisRBG, actionIndex]; %#ok<AGROW> % DRL MOD: Track index for i1
                    
                    % Track unique UEs for MaxNumUsersPerTTI
                    if ~ismember(selectedUE, totalUniqueUEsAllocated)
                        totalUniqueUEsAllocated = [totalUniqueUEsAllocated, selectedUE]; %#ok<AGROW>
                    end

                    if obj.DRLDebug
                        fprintf('[DRL] Layer %d, RBG %d: Allocated to UE %d\n', layer, rbg, selectedUE);
                    end
                end
                
                % ---------------------------------------------------------------
                % DRL MOD: Detailed per-RBG logging (when DRLDebug enabled)
                % ---------------------------------------------------------------
                if obj.DRLDebug && ~isempty(allocatedUEsThisRBG)
                    fprintf('[DRL Scheduler] RBG %d:\n', rbg);
                    
                    % Group UEs by their allocated layers
                    uniqueUEsOnRBG = unique(allocatedUEsThisRBG, 'stable');
                    for ueIdx = 1:numel(uniqueUEsOnRBG)
                        ue = uniqueUEsOnRBG(ueIdx);
                        ueLayersOnRBG = find(allocatedUEsThisRBG == ue) - 1;  % 0-indexed layers
                        numAllocatedLayers = ueLayerCount(ue, rbg);  % Layers allocated by scheduler
                        
                        % Get precoder info
                        ueEligibleIdx = find(eligibleUEs == ue, 1);
                        if ~isempty(ueEligibleIdx) && ueEligibleIdx <= numel(W)
                            W_ue = W{ueEligibleIdx};
                            if ~isempty(W_ue)
                                wSize = size(W_ue);
                                % W format: [rank x numPorts] or [rank x numPorts x numSubbands]
                                if numel(wSize) == 2
                                    wSizeStr = sprintf('[%d layers x %d ports]', wSize(1), wSize(2));
                                elseif numel(wSize) >= 3
                                    wSizeStr = sprintf('[%d layers x %d ports x %d subbands]', wSize(1), wSize(2), wSize(3));
                                else
                                    wSizeStr = sprintf('[%dx%d]', wSize);
                                end
                                fprintf('  UE%d: Layers=%d [%s], W%s\n', ...
                                    ue, numAllocatedLayers, strjoin(string(ueLayersOnRBG), ','), wSizeStr);
                            else
                                fprintf('  UE%d: Layers=%d [%s], W[empty]\n', ...
                                    ue, numAllocatedLayers, strjoin(string(ueLayersOnRBG), ','));
                            end
                        else
                            fprintf('  UE%d: Layers=%d [%s]\n', ...
                                ue, numAllocatedLayers, strjoin(string(ueLayersOnRBG), ','));
                        end
                    end
                    
                    % Check and log i1 match status
                    if obj.EnableI1Constraint && numel(uniqueUEsOnRBG) > 1
                        allI1Match = true;
                        for i = 1:numel(uniqueUEsOnRBG)-1
                            ue1Idx = find(eligibleUEs == uniqueUEsOnRBG(i), 1);
                            ue2Idx = find(eligibleUEs == uniqueUEsOnRBG(i+1), 1);
                            if ~isempty(ue1Idx) && ~isempty(ue2Idx) && ...
                               ue1Idx <= numel(ueI1Values) && ue2Idx <= numel(ueI1Values)
                                i1_1 = ueI1Values{ue1Idx};
                                i1_2 = ueI1Values{ue2Idx};
                                if ~isempty(i1_1) && ~isempty(i1_2) && ~all(i1_1 == i1_2)
                                    allI1Match = false;
                                    break;
                                end
                            end
                        end
                        fprintf('  i1 match: %s\n', ternary(allI1Match, 'YES', 'NO'));
                    end
                    
                    % Check and log orthogonality status
                    if obj.EnableOrthogonalityConstraint && numel(uniqueUEsOnRBG) > 1
                        maxCorrRBG = 0;
                        for i = 1:numel(uniqueUEsOnRBG)
                            for j = i+1:numel(uniqueUEsOnRBG)
                                ue1Idx = find(eligibleUEs == uniqueUEsOnRBG(i), 1);
                                ue2Idx = find(eligibleUEs == uniqueUEsOnRBG(j), 1);
                                if ~isempty(ue1Idx) && ~isempty(ue2Idx) && ...
                                   ue1Idx <= numel(W) && ue2Idx <= numel(W)
                                    W1 = W{ue1Idx};
                                    W2 = W{ue2Idx};
                                    if ~isempty(W1) && ~isempty(W2) && ~isreal(W1) && ~isreal(W2)
                                        try
                                            if ismatrix(W1) && ismatrix(W2)
                                                w1 = W1(:) / (norm(W1(:)) + 1e-10);
                                                w2 = W2(:) / (norm(W2(:)) + 1e-10);
                                                corrVal = abs(w1' * w2);
                                            else
                                                numSB = min(size(W1, 3), size(W2, 3));
                                                sbIdx = min(rbg, numSB);
                                                w1 = W1(:,:,sbIdx);
                                                w2 = W2(:,:,sbIdx);
                                                w1 = w1(:) / (norm(w1(:)) + 1e-10);
                                                w2 = w2(:) / (norm(w2(:)) + 1e-10);
                                                corrVal = abs(w1' * w2);
                                            end
                                            maxCorrRBG = max(maxCorrRBG, corrVal);
                                        catch
                                            % Skip on error
                                        end
                                    end
                                end
                            end
                        end
                        orthThreshold = 1 - obj.SemiOrthogonalityFactor;
                        orthPass = maxCorrRBG <= orthThreshold;
                        fprintf('  Orthogonality check: %s (corr=%.2f %s threshold %.2f)\n', ...
                            ternary(orthPass, 'PASS', 'FAIL'), maxCorrRBG, ...
                            ternary(orthPass, '<', '>'), orthThreshold);
                    end
                end
            end

            % ---------------------------------------------------------------
            % Extract allotted UEs (UEs with at least one RBG)
            % ---------------------------------------------------------------
            allottedUEsMask = any(ueRBGAllocation, 2);
            allottedUEsList = find(allottedUEsMask);

            % Filter to only include eligible UEs (in case of RNTI mismatch)
            allottedUEsList = intersect(allottedUEsList, eligibleUEs, 'stable');
            numAllottedUEs = numel(allottedUEsList);

            % ---------------------------------------------------------------
            % Build output arrays
            % ---------------------------------------------------------------
            allottedUEs = zeros(numAllottedUEs, 1);
            freqAllocation = zeros(numAllottedUEs, numRBGs);
            mcsIndex = zeros(numAllottedUEs, 1);
            W = schedulerInput.W;  % Use precomputed precoding from parent

            for idx = 1:numAllottedUEs
                rnti = allottedUEsList(idx);
                allottedUEs(idx) = rnti;
                freqAllocation(idx, :) = ueRBGAllocation(rnti, :);

                % Find index in original eligibleUEs
                ueIdxInEligible = find(eligibleUEs == rnti, 1);

                % Calculate MCS based on average CQI of allotted RBs
                % (Reusing logic from parent runSchedulingStrategyRAT0)
                gNBCarrierIndex = 1;
                allottedRBs = convertRBGBitmapToRBs(obj, rnti, gNBCarrierIndex, freqAllocation(idx,:));

                % Get channel quality for this UE
                cqiRB = schedulerInput.channelQuality(ueIdxInEligible, allottedRBs+1);
                cqiSetRB = floor(mean(cqiRB, 2));

                % Calculate DL MCS value using parent method
                mcsIndex(idx) = selectMCSIndexDL(obj, cqiSetRB, rnti);
                
                % ----------------------------------------------------------
                % DRL MOD: MCS Backoff for MU-MIMO transmissions
                % Reduce MCS when sharing RBG with other UEs to account for
                % residual inter-user interference (CSI-RS based has imperfect
                % interference suppression)
                % ----------------------------------------------------------
                if obj.MU_MCSBackoff > 0
                    % Count max co-scheduled UEs on any RBG allocated to this UE
                    myRBGs = find(freqAllocation(idx, :));
                    maxCoScheduled = 0;
                    for rbgCheck = myRBGs
                        numUEsOnRBG = sum(ueRBGAllocation(:, rbgCheck) > 0);
                        maxCoScheduled = max(maxCoScheduled, numUEsOnRBG);
                    end
                    % Apply backoff: reduce MCS by backoff * (numCoScheduled - 1)
                    if maxCoScheduled > 1
                        backoffAmount = obj.MU_MCSBackoff * (maxCoScheduled - 1);
                        mcsIndex(idx) = max(0, mcsIndex(idx) - backoffAmount);
                        if obj.DRLDebug
                            fprintf('[DRL] UE %d: MCS backoff %d (co-scheduled with %d UEs)\n', ...
                                rnti, backoffAmount, maxCoScheduled);
                        end
                    end
                end
            end

            % ---------------------------------------------------------------
            % Store allocation for debugging/logging
            % ---------------------------------------------------------------
            obj.LastDRLAllocation.allottedUEs = allottedUEs;
            obj.LastDRLAllocation.freqAllocation = freqAllocation;
            obj.LastDRLAllocation.layerAssignment = ueLayerCount;
            obj.LastDRLAllocation.eligibleUEs = eligibleUEs;

            if obj.DRLDebug
                fprintf('[nrDRLScheduler] DRL allocation complete: %d UEs allocated\n', numAllottedUEs);
                for idx = 1:numAllottedUEs
                    fprintf('  UE %d: %d RBGs, MCS=%d\n', allottedUEs(idx), ...
                        sum(freqAllocation(idx,:)), mcsIndex(idx));
                end
            end
        end
        % ===================================================================
        % DRL MOD END - DRL scheduling strategy
        % ===================================================================
    end

    % ===================================================================
    % DRL MOD START - Helper methods for DRL scheduling
    % ===================================================================
    methods (Access = public)
        function info = getDRLSchedulerInfo(obj)
            %getDRLSchedulerInfo Get information about DRL scheduler state
            %   Returns a struct with current DRL scheduler configuration and state

            info = struct();
            info.EnableDRL = obj.EnableDRL;
            info.NumLayers = obj.NumLayers;
            info.SchedulingStrategy = obj.SchedulingStrategy;
            info.MaxUsersPerRBG = obj.MaxUsersPerRBG;
            info.DRLActionSize = size(obj.DRLAction);
            info.HasDRLAction = ~isempty(obj.DRLAction);
            info.LastDRLAllocation = obj.LastDRLAllocation;

            % Get RBG info from cell config
            if ~isempty(obj.CellConfig)
                info.NumResourceBlocks = obj.CellConfig(1).NumResourceBlocks;
                if obj.NumUEs > 0 && ~isempty(obj.UEContext)
                    carrierContext = obj.UEContext(1).ComponentCarrier(1);
                    info.RBGSize = carrierContext.RBGSize;
                    info.NumRBGs = carrierContext.NumRBGs;
                end
            end
        end

        function valid = validateDRLAction(obj, action, numEligibleUEs, numRBGs)
            %validateDRLAction Validate DRL action matrix
            %   valid = validateDRLAction(obj, action, numEligibleUEs, numRBGs)
            %   Returns true if action matrix has valid dimensions and values

            valid = true;

            % Check dimensions
            [nL, nR] = size(action);
            if nL < 1 || nR < 1
                valid = false;
                if obj.DRLDebug
                    warning('DRL action has invalid dimensions: [%d x %d]', nL, nR);
                end
                return;
            end

            if nR ~= numRBGs
                valid = false;
                if obj.DRLDebug
                    warning('DRL action RBG count mismatch: %d vs expected %d', nR, numRBGs);
                end
            end

            % Check value range
            minVal = min(action(:));
            maxVal = max(action(:));
            expectedMax = numEligibleUEs + 1;

            if minVal < 1 || maxVal > expectedMax
                valid = false;
                if obj.DRLDebug
                    warning('DRL action values out of range: [%d, %d], expected [1, %d]', ...
                        minVal, maxVal, expectedMax);
                end
            end
        end

        function mask = computeActionMask(obj, eligibleUEs, freqOccupancyBitmap, currentAllocation)
            %computeActionMask Compute action mask for valid DRL actions
            %   mask = computeActionMask(obj, eligibleUEs, freqOccupancyBitmap, currentAllocation)
            %
            %   Returns a logical mask [NumLayers x NumRBGs x (NumEligibleUEs+1)]
            %   where mask(l,m,a) = true means action a is valid at layer l, RBG m
            %
            %   Masking rules:
            %   1. RBGs occupied by retransmissions: only "no allocation" action valid
            %   2. UE already allocated on RBG: that UE's action is invalid
            %   3. MaxUsersPerRBG reached: all UE allocation actions invalid

            numEligibleUEs = numel(eligibleUEs);
            numRBGs = numel(freqOccupancyBitmap);
            numActions = numEligibleUEs + 1;  % +1 for "no allocation"

            % Initialize mask: all actions valid by default
            mask = true(obj.NumLayers, numRBGs, numActions);

            for rbg = 1:numRBGs
                % Rule 1: RBG occupied by retransmission
                if freqOccupancyBitmap(rbg)
                    % Only "no allocation" action is valid
                    mask(:, rbg, 1:numEligibleUEs) = false;
                    continue;
                end

                % Check current allocation on this RBG
                if ~isempty(currentAllocation)
                    allocatedUEsThisRBG = find(currentAllocation(:, rbg) > 0);

                    % Count allocated UEs
                    numAllocated = numel(allocatedUEsThisRBG);

                    % Rule 3: MaxUsersPerRBG check
                    if numAllocated >= obj.MaxUsersPerRBG
                        mask(:, rbg, 1:numEligibleUEs) = false;
                        continue;
                    end

                    % Rule 2: Mask already allocated UEs
                    for ue = allocatedUEsThisRBG'
                        ueIdx = find(eligibleUEs == ue, 1);
                        if ~isempty(ueIdx)
                            mask(:, rbg, ueIdx) = false;
                        end
                    end
                end
            end
        end

        function state = getSchedulerState(obj, eligibleUEs, freqOccupancyBitmap)
            %getSchedulerState Get current scheduler state for DRL agent
            %   state = getSchedulerState(obj, eligibleUEs, freqOccupancyBitmap)
            %
            %   Returns a struct containing:
            %   - eligibleUEs: RNTIs of eligible UEs
            %   - numEligibleUEs: count of eligible UEs
            %   - numRBGs: number of RBGs
            %   - freqOccupancyBitmap: current RBG occupancy (1=occupied)
            %   - channelQuality: CQI per UE per RB
            %   - bufferStatus: buffer status per UE
            %   - rank: selected rank per UE
            %   - precoders: precoding matrices per UE

            state = struct();
            state.eligibleUEs = eligibleUEs;
            state.numEligibleUEs = numel(eligibleUEs);
            state.numRBGs = numel(freqOccupancyBitmap);
            state.freqOccupancyBitmap = freqOccupancyBitmap;
            state.numLayers = obj.NumLayers;

            % Gather per-UE information
            numEligibleUEs = numel(eligibleUEs);
            state.channelQuality = zeros(numEligibleUEs, obj.CellConfig(1).NumResourceBlocks);
            state.bufferStatus = zeros(numEligibleUEs, 1);
            state.rank = ones(numEligibleUEs, 1);

            for i = 1:numEligibleUEs
                rnti = eligibleUEs(i);
                ueCtx = obj.UEContext(rnti);
                carrierCtx = ueCtx.ComponentCarrier(1);

                % Channel quality
                csiMeasurement = carrierCtx.CSIMeasurementDL;
                if ~isempty(csiMeasurement.CSIRS.CQI)
                    cqiVal = max(csiMeasurement.CSIRS.CQI(:));
                    state.channelQuality(i, :) = cqiVal;
                    state.rank(i) = csiMeasurement.CSIRS.RI;
                end

                % Buffer status
                state.bufferStatus(i) = ueCtx.BufferStatusDL;
            end
        end
    end
    % ===================================================================
    % DRL MOD END - Helper methods
    % ===================================================================
end
