classdef SchedulerDRLAction < nrScheduler
    % SchedulerDRLAction
    % - Inherits almost all logic from nrScheduler
    % - Overrides only scheduleNewTransmissionsDL to use DRL actions
    % - Keeps retransmissions and base HARQ logic unchanged

    properties (Access = public)
        DRL_IP = "127.0.0.1";
        DRL_Port = 5555;
        DRL_Socket = [];
        DRL_IsConnected = false;

        DRL_RxBuf uint8 = uint8([]);
        DRL_Terminator = uint8(10);
        DRL_TimeoutSec = 5;

        MaxUEs = 16;
        SubbandSize = 16;
        MaxNumLayers = 16;
        PRGSize = [];

        % Metrics for DRL features
        AvgThroughputMBps = [];
        Rho = 0.9;
        LastServedBytes = [];
        LastAllocRatio = [];
        AvgAllocRatio = [];
        LastCSIRSReport = [];
        LastCSIRSUpdateTime = [];
        LastAllocationMatrix = [];
        LastSubbandCQI = [];
        LastRhoVec = [];
        LastPrecodingW = [];

        % Precoding adaptation (paper-style robustness)
        PrecodingAlphaMin = 0.3;
        PrecodingPowerGamma = 0.7;
        PrecodingCQIMin = 1;
        MU_MCS_BackoffPerUE = 2;
        MU_MCS_RhoSlope = 4;

        % Mapping Table: CQI to Spectral Efficiency (Approx)
        CQIToSE = [ ...
            0.0000, ...  % CQI 0
            0.1523, ...  % CQI 1
            0.2344, ...  % CQI 2
            0.3770, ...  % CQI 3
            0.6016, ...  % CQI 4
            0.8770, ...  % CQI 5
            1.1758, ...  % CQI 6
            1.4766, ...  % CQI 7
            1.9141, ...  % CQI 8
            2.4063, ...  % CQI 9
            2.7305, ...  % CQI 10
            3.3223, ...  % CQI 11
            3.9023, ...  % CQI 12
            4.5234, ...  % CQI 13
            5.1152, ...  % CQI 14
            5.5547  ...  % CQI 15
        ];
    end

    methods (Access = public)
        function obj = SchedulerDRLAction(varargin)
            %#ok<INUSD>
        end

        function success = connectToDRLAgent(obj)
            success = false;
            if ~isempty(obj.DRL_Socket)
                delete(obj.DRL_Socket);
                obj.DRL_Socket = [];
            end
            try
                disp('[MATLAB] Connecting to DRL Trainer ...');
                obj.DRL_Socket = tcpclient(obj.DRL_IP, obj.DRL_Port, ...
                    'Timeout', 60, 'ConnectTimeout', 60);
                obj.DRL_Socket.InputBufferSize = 1048576;  % 1MB
                obj.DRL_Socket.OutputBufferSize = 1048576; % 1MB
                obj.DRL_IsConnected = true;
                disp('[MATLAB] Connected!');
                success = true;
            catch
                disp('[MATLAB] Connection Failed.');
                obj.DRL_IsConnected = false;
            end
        end

        function drlSendJSON(obj, S)
            if ~obj.DRL_IsConnected
                error('[MATLAB] DRL not connected');
            end
            msg = jsonencode(S);
            data = [uint8(msg) obj.DRL_Terminator];  % append '\n'
            fwrite(obj.DRL_Socket, data);
        end

        function S = drlRecvJSON(obj, timeoutSec)
            if nargin < 3, timeoutSec = obj.DRL_TimeoutSec; end
            t0 = tic;

            while true
                nb = obj.DRL_Socket.BytesAvailable;
                if nb > 0
                    readSize = min(nb, 65536);
                    chunk = fread(obj.DRL_Socket, readSize);
                    obj.DRL_RxBuf = [obj.DRL_RxBuf; chunk(:)];
                end

                idx = find(obj.DRL_RxBuf == obj.DRL_Terminator, 1, 'first');
                if ~isempty(idx)
                    line = obj.DRL_RxBuf(1:idx-1);
                    obj.DRL_RxBuf = obj.DRL_RxBuf(idx+1:end);
                    if isempty(line)
                        continue;
                    end
                    S = jsondecode(char(line(:)'));
                    return;
                end

                if toc(t0) > timeoutSec
                    error('[MATLAB] Timeout waiting DRL JSON response');
                end
                pause(0.002);
            end
        end
    end

    methods (Access = protected)
        function dlAssignments = scheduleNewTransmissionsDL(obj, timeFrequencyResource, schedulingInfo)
            %#ok<INUSD>
            eligibleUEs = schedulingInfo.EligibleUEs;
            if isempty(eligibleUEs), dlAssignments = struct([]); return; end

            if ~isempty(obj.SchedulerConfig) && ~isempty(obj.SchedulerConfig.MUMIMOConfigDL)
                muCfg = obj.SchedulerConfig.MUMIMOConfigDL;
                if isfield(muCfg, 'MinCQI') && ~isempty(muCfg.MinCQI)
                    obj.PrecodingCQIMin = muCfg.MinCQI;
                end
            end

            if isempty(obj.AvgThroughputMBps)
                obj.AvgThroughputMBps = zeros(1, obj.MaxUEs);
                obj.LastServedBytes = zeros(1, obj.MaxUEs);
                obj.LastAllocRatio = zeros(1, obj.MaxUEs);
                obj.AvgAllocRatio = zeros(1, obj.MaxUEs);
                obj.LastCSIRSReport = cell(1, obj.MaxUEs);
                obj.LastCSIRSUpdateTime = nan(1, obj.MaxUEs);
                obj.LastSubbandCQI = cell(1, obj.MaxUEs);
                obj.LastRhoVec = cell(1, obj.MaxUEs);
                obj.LastPrecodingW = cell(1, obj.MaxUEs);
            end

            numRBs = obj.CellConfig.NumResourceBlocks;
            rbgSize = obj.getRBGSize();
            numRBGs = ceil(numRBs / rbgSize);
            numSubbands = ceil(numRBs / obj.SubbandSize);

            slotDur = obj.CellConfig(1).SlotDuration;
            currentTime = (double(obj.CurrFrame) * 10e-3) + (double(obj.CurrSlot) * slotDur * 1e-3);

            featDim = 5 + 2 * numSubbands;
            exportMatrix = zeros(obj.MaxUEs, featDim);
            precodingMatrixMap = cell(1, obj.MaxUEs);

            scheduledUEsBySubband = [];
            if ~isempty(obj.LastAllocationMatrix)
                scheduledUEsBySubband = obj.getScheduledUEsBySubband(obj.LastAllocationMatrix, numSubbands, rbgSize);
            end

            for i = 1:length(eligibleUEs)
                rnti = eligibleUEs(i);
                if rnti > obj.MaxUEs, continue; end

                ueCtx = obj.UEContext(rnti);
                carrierCtx = ueCtx.ComponentCarrier(1);
                csirsConfig = carrierCtx.CSIRSConfiguration;
                [dlRank, ~, wbCQI, sbCQI, W_out, ~] = obj.decodeCSIRS( ...
                    csirsConfig, currentTime, currentTime + 1e-3, carrierCtx, numRBs, rnti, currentTime);

                precodingMatrixMap{rnti} = W_out;
                obj.LastPrecodingW{rnti} = W_out;

                f_R = min(obj.AvgThroughputMBps(rnti) / 100, 1);
                f_h = dlRank / 2;
                f_d = obj.AvgAllocRatio(rnti);
                f_b = ueCtx.BufferStatusDL;

                cqiIdx = min(max(round(wbCQI), 0), 15);
                f_o = obj.CQIToSE(cqiIdx + 1) / 5.5547;

                if length(sbCQI) ~= numSubbands
                    sbCQI = ones(1, numSubbands) * wbCQI;
                end
                sb_cqi_idx = min(max(round(sbCQI), 0), 15);
                f_g_vec = obj.CQIToSE(sb_cqi_idx + 1) / 5.5547;

                if ~isempty(scheduledUEsBySubband)
                    f_rho_vec = obj.computeCrossCorrelation(precodingMatrixMap, scheduledUEsBySubband, rnti, numSubbands);
                else
                    scheduledUEs = eligibleUEs(1:i-1);
                    f_rho_vec = obj.computeCrossCorrelation(precodingMatrixMap, scheduledUEs, rnti, numSubbands);
                end

                obj.LastSubbandCQI{rnti} = sbCQI;
                obj.LastRhoVec{rnti} = f_rho_vec;

                exportMatrix(rnti, :) = [f_R, f_h, f_d, f_b, f_o, f_g_vec, f_rho_vec];
            end

            allocationMatrix = obj.communicateWithPythonTTI(exportMatrix, eligibleUEs, numSubbands, numRBs, numRBGs, rbgSize);
            % Minimal log: count non-zero allocations and unique UEs scheduled
            nonZero = nnz(allocationMatrix);
            uniqueUEs = unique(allocationMatrix(:));
            uniqueUEs = uniqueUEs(uniqueUEs > 0);
            fprintf('[DRL] alloc nnz=%d, uniqueUEs=%d\n', nonZero, numel(uniqueUEs));

            allocationMatrix = obj.applyPairingFilter( ...
                allocationMatrix, eligibleUEs, precodingMatrixMap, exportMatrix, rbgSize);

            [allottedUEs, freqAllocation, mcsIndex, W_final] = obj.applyAllocationMatrix( ...
                allocationMatrix, eligibleUEs, precodingMatrixMap, exportMatrix, rbgSize, schedulingInfo.MaxNumUsersTTI);

            servedBytes = zeros(1, obj.MaxUEs);
            currentAlloc = zeros(1, obj.MaxUEs);
            numRBGTotal = ceil(numRBs / rbgSize) * obj.MaxNumLayers;
            obj.AvgAllocRatio = obj.Rho * obj.AvgAllocRatio + (1-obj.Rho) * currentAlloc;
            for k = 1:length(allottedUEs)
                ueID = allottedUEs(k);
                numRBG = sum(freqAllocation(k,:));
                numLayers = obj.getNumLayersFromW(W_final{k});
                if numRBG > 0
                    mcs = mcsIndex(k);
                    bpp = obj.getBytesPerPRB(mcs);
                    servedBytes(ueID) = numRBG * rbgSize * bpp;
                    allocatedRBGs = numRBG * numLayers;
                    currentAlloc(ueID) = allocatedRBGs / numRBGTotal;
                end
            end
            obj.LastServedBytes = servedBytes;
            obj.LastAllocRatio = currentAlloc;
            instRateMbps = (servedBytes * 8) / 1e6;
            obj.AvgThroughputMBps = obj.Rho * obj.AvgThroughputMBps + (1 - obj.Rho) * instRateMbps;

            numAllotted = length(allottedUEs);
            dlAssignments = obj.DLGrantArrayStruct(1:numAllotted);
            for idx = 1:numAllotted
                selectedUE = allottedUEs(idx);
                dlAssignments(idx).RNTI = selectedUE;
                dlAssignments(idx).GNBCarrierIndex = 1;
                dlAssignments(idx).FrequencyAllocation = freqAllocation(idx, :);

                carrierCtx = obj.UEContext(selectedUE).ComponentCarrier(1);
                mcsOffset = fix(carrierCtx.MCSOffset(obj.DLType+1));
                rawMCS = mcsIndex(idx);
                finalMCS = min(max(rawMCS - mcsOffset, 0), 27);
                dlAssignments(idx).MCSIndex = finalMCS;

                Wcorr = W_final{idx};
                dlAssignments(idx).W = obj.formatPrecodingMatrix(Wcorr);
            end
        end
    end

    methods (Access = private)
        function allocationMatrix = communicateWithPythonTTI(obj, exportMatrix, eligibleUEs, numSubbands, numRBs, numRBGs, rbgSize)
            if ~obj.DRL_IsConnected
                error('[MATLAB] DRL not connected. Call connectToDRLAgent() first.');
            end

            payload = struct();
            payload.type = "TTI_OBS";
            payload.frame = double(obj.CurrFrame);
            payload.slot  = double(obj.CurrSlot);
            payload.max_ues = obj.MaxUEs;
            payload.max_layers = obj.MaxNumLayers;
            payload.max_layers_per_ue = 2;
            payload.num_rbg = numRBGs;
            payload.num_subbands = numSubbands;
            payload.num_rbs = numRBs;
            payload.rbg_size = rbgSize;
            payload.subband_size = obj.SubbandSize;
            payload.eligible_ues = eligibleUEs;
            payload.features = exportMatrix;
            if ~isempty(obj.SchedulerConfig) && ~isempty(obj.SchedulerConfig.MUMIMOConfigDL)
                muCfg = obj.SchedulerConfig.MUMIMOConfigDL;
                if isfield(muCfg, 'SemiOrthogonalityFactor')
                    payload.mu_corr_threshold = muCfg.SemiOrthogonalityFactor;
                end
                if isfield(muCfg, 'MinCQI')
                    payload.min_cqi = muCfg.MinCQI;
                end
            end

            obj.drlSendJSON(payload);
            resp = obj.drlRecvJSON(obj.DRL_TimeoutSec);

            if ~isfield(resp, "type") || resp.type ~= "TTI_ALLOC"
                error("[MATLAB] Invalid response type");
            end
            if ~isfield(resp, "allocation")
                error("[MATLAB] Missing allocation in response");
            end

            allocationMatrix = resp.allocation;
            if ~isequal(size(allocationMatrix), [numRBGs, obj.MaxNumLayers])
                error("[MATLAB] allocation size mismatch. Got %s, expected [%d %d].", ...
                    mat2str(size(allocationMatrix)), numRBGs, obj.MaxNumLayers);
            end

            allocationMatrix(~ismember(allocationMatrix, [0 eligibleUEs])) = 0;
            obj.LastAllocationMatrix = allocationMatrix;
        end

        function [finalUEs, finalFreqAlloc, finalMCS, finalW] = applyAllocationMatrix(obj, allocationMatrix, eligibleUEs, pMap, feats, rbgSize, maxUsers)
            [numRBGs, numLayers] = size(allocationMatrix);
            numRBs = obj.CellConfig.NumResourceBlocks;
            numSubbands = ceil(numRBs / obj.SubbandSize);
            corrThreshold = 0.9;
            if ~isempty(obj.SchedulerConfig) && ~isempty(obj.SchedulerConfig.MUMIMOConfigDL)
                muCfg = obj.SchedulerConfig.MUMIMOConfigDL;
                if isfield(muCfg, 'SemiOrthogonalityFactor') && ~isempty(muCfg.SemiOrthogonalityFactor)
                    corrThreshold = muCfg.SemiOrthogonalityFactor;
                end
            end

            tempFreqAlloc = zeros(obj.MaxUEs, numRBGs);
            tempMCS = zeros(obj.MaxUEs, 1);
            tempW = cell(obj.MaxUEs, 1);

            for rbg = 1:numRBGs
                scheduledUEsOnRBG = [];
                for layer = 1:numLayers
                    ue = allocationMatrix(rbg, layer);
                    if ue > 0 && ue <= obj.MaxUEs && ismember(ue, eligibleUEs)
                        tempFreqAlloc(ue, rbg) = 1;
                        if ~ismember(ue, scheduledUEsOnRBG)
                            scheduledUEsOnRBG = [scheduledUEsOnRBG, ue];
                        end
                    end
                end

                if ~isempty(scheduledUEsOnRBG)
                    for ue = scheduledUEsOnRBG
                        if ue <= length(pMap) && ~isempty(pMap{ue}) && isempty(tempW{ue})
                            tempW{ue} = pMap{ue};
                        end
                    end
                end
            end

            % Pairing count per RBG (for MU-MIMO backoff)
            pairedCountPerRBG = zeros(1, numRBGs);
            for rbg = 1:numRBGs
                ueList = unique(allocationMatrix(rbg, :));
                ueList = ueList(ueList > 0);
                pairedCountPerRBG(rbg) = numel(ueList);
            end

            for ue = 1:obj.MaxUEs
                if any(tempFreqAlloc(ue, :) > 0)
                    % Effective CQI from allocated subbands (robust)
                    rbgIdxs = find(tempFreqAlloc(ue, :) > 0);
                    sbIdxs = unique(ceil(((rbgIdxs - 1) * rbgSize + 1) / obj.SubbandSize));
                    sbIdxs = min(max(sbIdxs, 1), numSubbands);

                    cqi_eff = [];
                    if ue <= numel(obj.LastSubbandCQI) && ~isempty(obj.LastSubbandCQI{ue})
                        sbCQI = obj.LastSubbandCQI{ue};
                        sbCQI = sbCQI(1:min(numel(sbCQI), numSubbands));
                        sbCQI = sbCQI(sbIdxs);
                        if ~isempty(sbCQI)
                            cqi_eff = floor(prctile(sbCQI, 25)); % robust lower quartile
                        end
                    end
                    if isempty(cqi_eff) || ~isfinite(cqi_eff)
                        cqi_eff = obj.getWBcqifromFeatsOrCSI(ue, feats);
                    end
                    cqi_eff = min(max(round(cqi_eff), 1), 15);
                    mcs = getMCSIndex(obj, cqi_eff);

                    % MU-MIMO backoff based on pairing count
                    pairedMax = max(pairedCountPerRBG(rbgIdxs));
                    backoff = max(pairedMax - 1, 0) * obj.MU_MCS_BackoffPerUE;

                    % Additional backoff if correlation high
                    rho_max = 0;
                    if ue <= numel(obj.LastRhoVec) && ~isempty(obj.LastRhoVec{ue})
                        rhoVec = obj.LastRhoVec{ue};
                        rhoVec = rhoVec(1:min(numel(rhoVec), numSubbands));
                        rho_max = max(rhoVec(sbIdxs));
                    end
                    if rho_max > corrThreshold
                        backoff = backoff + ceil((rho_max - corrThreshold) * obj.MU_MCS_RhoSlope);
                    end

                    tempMCS(ue) = max(mcs - backoff, 0);
                end
            end

            scheduledUEs = find(sum(tempFreqAlloc, 2) > 0);
            if length(scheduledUEs) > maxUsers
                allocCounts = sum(tempFreqAlloc(scheduledUEs, :), 2);
                [~, sortIdx] = sort(allocCounts, 'descend');
                scheduledUEs = scheduledUEs(sortIdx(1:maxUsers));
            end

            finalUEs = scheduledUEs';
            finalFreqAlloc = tempFreqAlloc(scheduledUEs, :);
            finalMCS = tempMCS(scheduledUEs);
            finalW = cell(numel(scheduledUEs), 1);
            for idx = 1:numel(scheduledUEs)
                ue = scheduledUEs(idx);
                Wbase = [];
                if ue <= numel(tempW)
                    Wbase = tempW{ue};
                end
                finalW{idx} = obj.buildSubbandPrecoding(ue, Wbase, finalFreqAlloc(idx, :), rbgSize);
            end
        end

        function Wout = buildSubbandPrecoding(obj, ue, Wbase, rbgAlloc, rbgSize)
            if isempty(Wbase)
                Wout = Wbase;
                return
            end

            numRBs = obj.CellConfig.NumResourceBlocks;
            numSubbands = ceil(numRBs / obj.SubbandSize);
            alphaMin = obj.PrecodingAlphaMin;
            gamma = obj.PrecodingPowerGamma;
            minCQI = obj.PrecodingCQIMin;

            if ndims(Wbase) >= 3
                numPRG = size(Wbase, 3);
                prgSize = ceil(numRBs / numPRG);
            else
                numPRG = numSubbands;
                prgSize = obj.SubbandSize;
                Wbase = repmat(Wbase, 1, 1, numPRG);
            end

            % Wideband robust precoder
            Wwb = mean(Wbase, 3);
            Wout = Wbase;
            powerWeights = ones(1, numPRG);

            sbCQI = [];
            rhoVec = [];
            if ue <= numel(obj.LastSubbandCQI)
                sbCQI = obj.LastSubbandCQI{ue};
            end
            if ue <= numel(obj.LastRhoVec)
                rhoVec = obj.LastRhoVec{ue};
            end

            for prg = 1:numPRG
                rbStart = (prg - 1) * prgSize + 1;
                sbIdx = ceil(rbStart / obj.SubbandSize);
                sbIdx = min(max(sbIdx, 1), numSubbands);

                rho = 0;
                if ~isempty(rhoVec) && numel(rhoVec) >= sbIdx
                    rho = rhoVec(sbIdx);
                end
                cqi = minCQI;
                if ~isempty(sbCQI) && numel(sbCQI) >= sbIdx
                    cqi = sbCQI(sbIdx);
                end
                cqi = min(max(round(cqi), 0), 15);
                se = obj.CQIToSE(cqi + 1);

                alpha = 1 - rho;
                alpha = min(max(alpha, alphaMin), 1.0);
                if cqi < minCQI
                    alpha = 0.0;
                end

                Wcb = Wbase(:, :, prg);
                Wrel = alpha * Wcb + (1 - alpha) * Wwb;

                % Normalize per-layer
                for l = 1:size(Wrel, 1)
                    w = Wrel(l, :);
                    nrm = norm(w);
                    if nrm > 0
                        Wrel(l, :) = w / nrm;
                    end
                end
                Wout(:, :, prg) = Wrel;

                % Power rebalancing weight from subband SE
                powerWeights(prg) = max(se, 0.1) ^ gamma;
            end

            % Apply power weights and renormalize average power
            meanPow = mean(powerWeights);
            if meanPow <= 0
                meanPow = 1.0;
            end
            powerWeights = powerWeights / meanPow;
            for prg = 1:numPRG
                Wout(:, :, prg) = Wout(:, :, prg) * sqrt(powerWeights(prg));
            end
        end

        function allocationMatrix = applyPairingFilter(obj, allocationMatrix, eligibleUEs, precodingMap, feats, rbgSize)
            if isempty(allocationMatrix)
                return
            end
            if isempty(obj.SchedulerConfig) || isempty(obj.SchedulerConfig.MUMIMOConfigDL)
                return
            end
            muCfg = obj.SchedulerConfig.MUMIMOConfigDL;
            if isfield(muCfg, 'MaxNumUsersPaired')
                maxUsersPaired = muCfg.MaxNumUsersPaired;
            else
                maxUsersPaired = obj.MaxNumLayers;
            end
            if isfield(muCfg, 'SemiOrthogonalityFactor')
                corrThreshold = muCfg.SemiOrthogonalityFactor;
            else
                corrThreshold = 0.9;
            end
            if isfield(muCfg, 'MinCQI')
                minCQI = muCfg.MinCQI;
            else
                minCQI = 1;
            end

            numRBs = obj.CellConfig.NumResourceBlocks;
            numSubbands = ceil(numRBs / obj.SubbandSize);
            [numRBGs, ~] = size(allocationMatrix);

            for rbg = 1:numRBGs
                % Cap number of layers per UE on this RBG based on reported rank
                ueListRaw = unique(allocationMatrix(rbg, :));
                ueListRaw = ueListRaw(ueListRaw > 0);
                for k = 1:numel(ueListRaw)
                    ue = ueListRaw(k);
                    W = [];
                    if ue <= numel(precodingMap)
                        W = precodingMap{ue};
                    end
                    rank = obj.getNumLayersFromW(W);
                    if rank < 1
                        rank = 1;
                    end
                    layerIdx = find(allocationMatrix(rbg, :) == ue);
                    if numel(layerIdx) > rank
                        allocationMatrix(rbg, layerIdx(rank+1:end)) = 0;
                    end
                end

                ueList = unique(allocationMatrix(rbg, :));
                ueList = ueList(ueList > 0);
                if isempty(ueList)
                    continue
                end
                ueList = ueList(ismember(ueList, eligibleUEs));
                if numel(ueList) <= 1
                    continue
                end

                rbStart = (rbg - 1) * rbgSize + 1;
                subbandIdx = ceil(rbStart / obj.SubbandSize);
                subbandIdx = min(max(subbandIdx, 1), numSubbands);

                metrics = -inf(1, numel(ueList));
                for k = 1:numel(ueList)
                    ue = ueList(k);
                    [sbCQI, sbSE] = obj.getSubbandCQIFromFeatsOrCSI(ue, feats, subbandIdx);
                    if sbCQI < minCQI
                        continue
                    end
                    buf = 0;
                    if ue <= numel(obj.UEContext)
                        buf = obj.UEContext(ue).BufferStatusDL;
                    end
                    if buf <= 0
                        continue
                    end
                    metrics(k) = obj.getPFMetricSubband(ue, sbSE) + (1e-6 * buf);
                end

                validMask = isfinite(metrics);
                if ~any(validMask)
                    allocationMatrix(rbg, :) = 0;
                    continue
                end

                [~, primaryIdx] = max(metrics);
                selectedUEs = ueList(primaryIdx);
                remainingUEs = ueList;
                remainingUEs(remainingUEs == selectedUEs) = [];

                while numel(selectedUEs) < maxUsersPaired && ~isempty(remainingUEs)
                    bestUE = 0;
                    bestRho = inf;
                    bestMetric = -inf;
                    for k = 1:numel(remainingUEs)
                        ue = remainingUEs(k);
                        idxInList = find(ueList == ue, 1);
                        if isempty(idxInList) || ~isfinite(metrics(idxInList))
                            continue
                        end
                        W1 = precodingMap{ue};
                        if isempty(W1)
                            continue
                        end
                        rho = 0;
                        for s = 1:numel(selectedUEs)
                            ue2 = selectedUEs(s);
                            W2 = precodingMap{ue2};
                            if isempty(W2)
                                rho = inf;
                                break
                            end
                            rho = max(rho, obj.precoderPairCorrelation(W1, W2, subbandIdx));
                            if rho > corrThreshold
                                break
                            end
                        end
                        if rho <= corrThreshold
                            if (rho < bestRho) || (abs(rho - bestRho) < 1e-6 && metrics(idxInList) > bestMetric)
                                bestUE = ue;
                                bestRho = rho;
                                bestMetric = metrics(idxInList);
                            end
                        end
                    end
                    if bestUE == 0
                        break
                    end
                    selectedUEs = [selectedUEs bestUE]; %#ok<AGROW>
                    remainingUEs(remainingUEs == bestUE) = [];
                end
                if ~isempty(selectedUEs)
                    mask = ismember(allocationMatrix(rbg, :), selectedUEs);
                    allocationMatrix(rbg, ~mask) = 0;
                end
            end
        end

        function cqi = getWBcqifromFeatsOrCSI(obj, ue, feats)
            cqi = 7;
            if ue <= numel(obj.UEContext)
                carrierCtx = obj.UEContext(ue).ComponentCarrier(1);
                if ~isempty(carrierCtx.CSIMeasurementDL) && ~isempty(carrierCtx.CSIMeasurementDL.CSIRS)
                    cqi = carrierCtx.CSIMeasurementDL.CSIRS.CQI(1);
                end
            end
            if isempty(cqi) || ~isfinite(cqi)
                if ue <= size(feats, 1) && size(feats, 2) >= 5
                    cqi = round(feats(ue, 5) * 15);
                else
                    cqi = 7;
                end
            end
            cqi = min(max(round(cqi), 0), 15);
        end

        function [cqi, se] = getSubbandCQIFromFeatsOrCSI(obj, ue, feats, subbandIdx)
            cqi = obj.getWBcqifromFeatsOrCSI(ue, feats);
            se = obj.CQIToSE(cqi + 1);
            if ue <= size(feats, 1)
                sbIndex = 5 + subbandIdx;
                if sbIndex <= size(feats, 2)
                    sbSE = feats(ue, sbIndex) * 5.5547;
                    if sbSE > 0
                        [~, idx] = min(abs(obj.CQIToSE - sbSE));
                        cqi = idx - 1;
                        se = sbSE;
                    end
                end
            end
        end

        function metric = getPFMetricSubband(obj, ue, se)
            if se <= 0
                se = 0.1523; % CQI 1 fallback
            end
            avgRate = 1;
            if ~isempty(obj.AvgThroughputMBps) && ue <= numel(obj.AvgThroughputMBps)
                avgRate = obj.AvgThroughputMBps(ue);
                if ~isfinite(avgRate) || avgRate <= 0
                    avgRate = 1e-3;
                end
            end
            metric = se / avgRate;
        end

        function csiReport = getLatestCSIRSReport(obj, rnti, carrierCtx)
            rawReport = [];
            csiReport = struct();

            if ~isempty(carrierCtx) && isfield(carrierCtx, 'CSIMeasurementDL') && ...
                    isfield(carrierCtx.CSIMeasurementDL, 'CSIRS')
                rawReport = carrierCtx.CSIMeasurementDL.CSIRS;
            end

            if isempty(rawReport)
                if rnti <= numel(obj.UEContext)
                    ueCtx = obj.UEContext(rnti);
                    if isfield(ueCtx, 'CSIMeasurementDL') && isfield(ueCtx.CSIMeasurementDL, 'CSIRS')
                        rawReport = ueCtx.CSIMeasurementDL.CSIRS;
                    end
                end
            end

            if ~isempty(rawReport)
                csiReport = rawReport;
            end

            if rnti <= numel(obj.LastCSIRSReport) && ~isempty(obj.LastCSIRSReport{rnti})
                cached = obj.LastCSIRSReport{rnti};
                if (~isfield(csiReport, 'W') || isempty(csiReport.W)) && ...
                        isfield(cached, 'W') && ~isempty(cached.W)
                    csiReport.W = cached.W;
                end
                if (~isfield(csiReport, 'CQI') || isempty(csiReport.CQI)) && ...
                        isfield(cached, 'CQI') && ~isempty(cached.CQI)
                    csiReport.CQI = cached.CQI;
                end
                if (~isfield(csiReport, 'RI') || isempty(csiReport.RI)) && ...
                        isfield(cached, 'RI') && ~isempty(cached.RI)
                    csiReport.RI = cached.RI;
                end
                if (~isfield(csiReport, 'PMISet') || isempty(csiReport.PMISet)) && ...
                        isfield(cached, 'PMISet') && ~isempty(cached.PMISet)
                    csiReport.PMISet = cached.PMISet;
                end
            end
        end

        function [dlRank, pmiSet, widebandCQI, cqiSubband, precodingMatrix, sinrEffSubband] = ...
                decodeCSIRS(obj, csirsConfig, pktStartTime, pktEndTime, carrierCtx, numRBs, rnti, currentTime)
            %#ok<INUSD>
            numTx = obj.CellConfig.NumTransmitAntennas;
            numSB = ceil(numRBs / obj.SubbandSize);

            csiReport = obj.getLatestCSIRSReport(rnti, carrierCtx);
            hasNewCSIRS = ~isempty(csiReport) && ( ...
                (isfield(csiReport,'W') && ~isempty(csiReport.W)) || ...
                (isfield(csiReport,'CQI') && ~isempty(csiReport.CQI)) || ...
                (isfield(csiReport,'RI') && ~isempty(csiReport.RI)) );

            if ~hasNewCSIRS
                dlRank = 1;
                pmiSet = [];
                widebandCQI = 0;
                cqiSubband = zeros(1, numSB);
                precodingMatrix = ones(1, numTx) ./ sqrt(numTx);
                sinrEffSubband = [];
                return
            end

            if hasNewCSIRS && (isempty(obj.LastCSIRSReport{rnti}) || ~isequaln(csiReport, obj.LastCSIRSReport{rnti}))
                obj.LastCSIRSReport{rnti} = csiReport;
                obj.LastCSIRSUpdateTime(rnti) = currentTime;
            else
                csiReport = obj.LastCSIRSReport{rnti};
            end

            rawCQI = csiReport.CQI;
            if isempty(rawCQI) || all(isnan(rawCQI(:)))
                widebandCQI = 0;
                cqiSubband = zeros(1, numSB);
            elseif isscalar(rawCQI)
                widebandCQI = rawCQI;
                cqiSubband = ones(1, numSB) * rawCQI;
            else
                widebandCQI = mean(rawCQI, 'all');
                cqiSubband = rawCQI(:).';
                if length(cqiSubband) ~= numSB
                    cqiSubband = imresize(cqiSubband, [1 numSB], 'nearest');
                end
            end
            widebandCQI = min(max(round(widebandCQI), 0), 15);
            cqiSubband  = min(max(round(cqiSubband),  0), 15);

            if isfield(csiReport,'RI') && ~isempty(csiReport.RI)
                dlRank = csiReport.RI;
            else
                dlRank = 1;
            end

            if isfield(csiReport,'W') && ~isempty(csiReport.W)
                W_raw = csiReport.W;
                if ndims(W_raw) == 3
                    W3D = W_raw;
                    K = size(W3D,3);
                    if K < numSB
                        W3D = cat(3, W3D, repmat(W3D(:,:,end), 1, 1, numSB-K));
                    elseif K > numSB
                        W3D = W3D(:,:,1:numSB);
                    end
                else
                    W2 = W_raw;
                    if size(W2,1) == dlRank && size(W2,2) == numTx
                        W2 = W2.'; 
                    end
                    if ~(size(W2,1)==numTx && size(W2,2)==dlRank)
                        W2 = ones(numTx, dlRank) ./ sqrt(numTx);
                    end
                    W3D = repmat(W2, 1, 1, numSB);
                end
            else
                W2  = ones(numTx, dlRank) ./ sqrt(numTx);
                W3D = repmat(W2, 1, 1, numSB);
            end

            precodingMatrix = W3D;
            pmiSet = [];
            sinrEffSubband = [];
        end

        function rbgSize = getRBGSize(obj)
            numRBs = obj.CellConfig.NumResourceBlocks;
            if numRBs <= 36, rbgSize = 2; elseif numRBs <= 72, rbgSize = 4;
            elseif numRBs <= 144, rbgSize = 8; else, rbgSize = 16; end
        end

        function bpp = getBytesPerPRB(~, mcs)
            effs = [0.15 0.23 0.38 0.60 0.88 1.18 1.48 1.91 2.40 2.73 3.32 3.90 4.52 5.12 5.55 6.07 6.23 6.50 6.70 6.90 7.00 7.10 7.20 7.30 7.35 7.40 7.45 7.48 7.50];
            if mcs<0,mcs=0;end; if mcs>28,mcs=28;end
            bpp = (effs(mcs+1) * 12 * 14 * 0.9) / 8;
        end

        function rho_vec = computeCrossCorrelation(obj, precodingMap, scheduledUEs, rnti, numSubbands)
            rho_vec = zeros(1, numSubbands);
            if rnti > numel(precodingMap)
                return
            end
            candidateW = precodingMap{rnti};
            if isempty(candidateW)
                return
            end

            for m = 1:numSubbands
                if iscell(scheduledUEs)
                    ueList = scheduledUEs{m};
                else
                    ueList = scheduledUEs;
                end
                if isempty(ueList)
                    continue
                end
                ueList = ueList(ueList ~= rnti);
                maxCorr = 0;
                for idx = 1:numel(ueList)
                    otherUE = ueList(idx);
                    if otherUE > numel(precodingMap)
                        continue
                    end
                    otherW = precodingMap{otherUE};
                    if isempty(otherW)
                        continue
                    end
                    kappa = obj.precoderPairCorrelation(candidateW, otherW, m);
                    maxCorr = max(maxCorr, kappa);
                end
                rho_vec(m) = maxCorr;
            end
        end

        function scheduledUEsBySubband = getScheduledUEsBySubband(obj, allocationMatrix, numSubbands, rbgSize)
            scheduledUEsBySubband = cell(1, numSubbands);
            if isempty(allocationMatrix)
                return
            end
            [numRBGs, ~] = size(allocationMatrix);
            for rbg = 1:numRBGs
                ueList = unique(allocationMatrix(rbg, :));
                ueList = ueList(ueList > 0);
                if isempty(ueList)
                    continue
                end
                rbStart = (rbg - 1) * rbgSize + 1;
                subbandIdx = ceil(rbStart / obj.SubbandSize);
                subbandIdx = min(max(subbandIdx, 1), numSubbands);
                scheduledUEsBySubband{subbandIdx} = unique([scheduledUEsBySubband{subbandIdx}, ueList]);
            end
        end

        function Pm = getPrecodingSubband(obj, W, subbandIdx)
            if isempty(W)
                Pm = [];
                return
            end
            if isscalar(W)
                Pm = W;
                return
            end
            if ndims(W) >= 3
                maxIdx = size(W, 3);
                numRBs = obj.CellConfig.NumResourceBlocks;
                numSubbands = ceil(numRBs / obj.SubbandSize);
                if maxIdx ~= numSubbands
                    prgSize = ceil(numRBs / maxIdx);
                    rbStart = (subbandIdx - 1) * obj.SubbandSize + 1;
                    prgIdx = ceil(rbStart / prgSize);
                    prgIdx = min(max(prgIdx, 1), maxIdx);
                    Pm = W(:, :, prgIdx);
                else
                    subbandIdx = min(subbandIdx, maxIdx);
                    Pm = W(:, :, subbandIdx);
                end
                return
            end
            Pm = W;
        end

        function corr = precoderPairCorrelation(obj, W1, W2, subbandIdx)
            if isempty(W1) || isempty(W2)
                corr = 0.0;
                return
            end
            P1 = obj.getPrecodingSubband(W1, subbandIdx);
            P2 = obj.getPrecodingSubband(W2, subbandIdx);
            if isempty(P1) || isempty(P2)
                corr = 0.0;
                return
            end
            corrMatrix = P1' * P2;
            colSum = abs(sum(corrMatrix, 1));
            corr = max(colSum);
        end

        function numLayers = getNumLayersFromW(~, W)
            if isempty(W)
                numLayers = 1;
                return
            end
            if isnumeric(W)
                if isscalar(W)
                    numLayers = 1;
                else
                    sz = size(W);
                    if ndims(W) >= 3
                        numLayers = sz(2);
                    else
                        if sz(1) == 32
                            numLayers = sz(2);
                        else
                            numLayers = sz(1);
                        end
                    end
                end
                return
            end
            numLayers = 1;
        end

        function W_proper = formatPrecodingMatrix(~, W_raw)
            if isempty(W_raw) || isscalar(W_raw)
                W_proper = ones(1, 32) / sqrt(32);
                return
            end
            sz = size(W_raw);
            if sz(1) >= 1 && sz(1) <= 8 && sz(2) == 32
                W_proper = W_raw;
                return
            end
            if sz(1) == 32 && sz(2) >= 1 && sz(2) <= 8
                if ndims(W_raw) == 2
                    W_proper = W_raw.';
                else
                    W_proper = permute(W_raw, [2, 1, 3]);
                end
                return
            end
            W_proper = ones(1, 32) / sqrt(32);
        end
    end
end
