classdef SchedulerDRL < nrScheduler
    % SchedulerDRL - Wideband Precoding Version
    % - Follows nrScheduler contract for DL grants:
    %   dlAssignments(i).W must be RANK-by-P-by-1 (wideband: single PRG)
    % - Uses wideband precoding (1 PRG spanning entire bandwidth)
    % - MU pairing filter uses wideband kappa correlation

    properties (Access = public)
        % --- DRL comm ---
        DRL_IP = "127.0.0.1";
        DRL_Port = 5555;
        DRL_Socket = [];
        DRL_IsConnected = false;
        DRL_RxBuf uint8 = uint8([]);
        DRL_Terminator = uint8(10);
        DRL_TimeoutSec = 5;

        % --- System limits ---
        MaxUEs = 16;
        MaxNumLayers = 16;
        MaxUsersPerRBG = 2;        % conservative to reduce BLER
        SubbandSize = 16;          % RB per "feature subband" (CQI feature)

        % --- Paper-style MU control ---
        MU_CorrThreshold = 1;    % threshold on kappa (tune)
        MU_CorrLog = true;

        % --- MCS control (WIDEBAND needs more conservative settings) ---
        % BLER was 40-60% with old settings, increase backoff significantly
        MU_MCSBackoffMax = 10;      % Increased from 6 (max backoff for high correlation)
        MU_MCSBackoffMin = 3;       % Increased from 0 (baseline backoff for wideband)
        MU_MCSBackoffStartRho = 0.1;  % Start backoff earlier
        MU_MCSBackoffFullRho  = 0.5;  % Reach max backoff at lower correlation
        
        % Additional wideband backoff (applied on top of MU backoff)
        WidebandMCSBackoff = 2;     % Extra backoff for wideband precoding

        % --- Metrics ---
        RhoEWMA = 0.9;
        AvgThroughputMbps = [];
        LastServedBytes = [];
        LastAllocRatio = [];
        AvgAllocRatio = [];
        LastCSIRSReport = [];
        LastCSIRSUpdateTime = [];
        LastAllocationMatrix = [];

        CQIToSE = [ ...
            0.0000,0.1523,0.2344,0.3770,0.6016,0.8770,1.1758,1.4766, ...
            1.9141,2.4063,2.7305,3.3223,3.9023,4.5234,5.1152,5.5547];
    end

    methods (Access = public)

        function obj = SchedulerDRL(varargin) %#ok<INUSD>
        end

        function success = connectToDRLAgent(obj)
            success = false;
            if ~isempty(obj.DRL_Socket)
                try delete(obj.DRL_Socket); catch, end
                obj.DRL_Socket = [];
            end
            try
                obj.DRL_Socket = tcpclient(obj.DRL_IP, obj.DRL_Port, ...
                    'Timeout', 60, 'ConnectTimeout', 60);
                obj.DRL_Socket.InputBufferSize  = 1048576;
                obj.DRL_Socket.OutputBufferSize = 1048576;
                obj.DRL_IsConnected = true;
                disp('[MATLAB] Connected to Python DRL!');
                success = true;
            catch ME
                disp('[MATLAB] DRL connect failed.');
                disp(ME.message);
                obj.DRL_IsConnected = false;
            end
        end

        function updateChannelQualityDL(obj, channelQualityInfo)
            % Cache CSI-RS report for scheduler-side use
            updateChannelQualityDL@nrScheduler(obj, channelQualityInfo);
            rnti = channelQualityInfo.RNTI;
            if rnti <= obj.MaxUEs
                slotDur = obj.CellConfig(1).SlotDuration;
                now = (double(obj.CurrFrame) * 10e-3) + (double(obj.CurrSlot) * slotDur * 1e-3);

                obj.LastCSIRSUpdateTime(rnti) = now;
                csiReport = struct();
                csiReport.RI = channelQualityInfo.RankIndicator;
                csiReport.CQI = channelQualityInfo.CQI;
                csiReport.W = channelQualityInfo.W;         % often P x RI x NSB
                if isfield(channelQualityInfo, 'PMISet')
                    csiReport.PMISet = channelQualityInfo.PMISet;
                end
                obj.LastCSIRSReport{rnti} = csiReport;
            end
        end
    end

    % =========================
    % ====== MAIN DL ==========
    % =========================
    methods (Access = protected)

        function dlAssignments = scheduleNewTransmissionsDL(obj, timeFrequencyResource, schedulingInfo) %#ok<INUSD>
            eligibleUEs = schedulingInfo.EligibleUEs;
            if isempty(eligibleUEs)
                dlAssignments = struct([]);
                return;
            end

            % ---- metrics init ----
            if isempty(obj.AvgThroughputMbps)
                obj.AvgThroughputMbps = zeros(1, obj.MaxUEs);
                obj.LastServedBytes   = zeros(1, obj.MaxUEs);
                obj.LastAllocRatio    = zeros(1, obj.MaxUEs);
                obj.AvgAllocRatio     = zeros(1, obj.MaxUEs);
                obj.LastCSIRSReport   = cell(1, obj.MaxUEs);
                obj.LastCSIRSUpdateTime = nan(1, obj.MaxUEs);
            end

            % ---- geometry ----
            carrierIndex = 1;
            numRBs = obj.CellConfig(carrierIndex).NumResourceBlocks;
            carrierCtx0 = obj.UEContext(eligibleUEs(1)).ComponentCarrier(carrierIndex);

            rbgSize = carrierCtx0.RBGSize;                         % use nrScheduler carrierCtx
            numRBGs = ceil(numRBs / rbgSize);

            % Feature subbands (for CQI feature only)
            numSubbandsFeat = ceil(numRBs / obj.SubbandSize);

            slotDur = obj.CellConfig(carrierIndex).SlotDuration;
            now = (double(obj.CurrFrame) * 10e-3) + (double(obj.CurrSlot) * slotDur * 1e-3);

            % ---- gather CSI per UE ----
            featDim = 5 + 2 * numSubbandsFeat;
            exportMatrix = zeros(obj.MaxUEs, featDim);

            % WIDEBAND precoders: Wprg{u} is [RI x P x 1] (single PRG)
            WprgMap = cell(1, obj.MaxUEs);

            % For MU kappa we use WprgMap (wideband)
            % For MCS: use subband CQI feature map
            sbCQIMap = nan(obj.MaxUEs, numSubbandsFeat);
            wbCQIMap = nan(1, obj.MaxUEs);
            rankMap  = ones(1, obj.MaxUEs);

            scheduledUEsWideband = [];
            if ~isempty(obj.LastAllocationMatrix)
                % WIDEBAND: Build scheduled UEs list (single PRG)
                scheduledUEsWideband = obj.getScheduledUEsWideband(obj.LastAllocationMatrix);
            end

            for k = 1:numel(eligibleUEs)
                rnti = eligibleUEs(k);
                if rnti > obj.MaxUEs, continue; end

                ueCtx = obj.UEContext(rnti);
                carrierCtx = ueCtx.ComponentCarrier(carrierIndex);

                [dlRank, wbCQI, sbCQI_feat, Wprg] = obj.decodeCSIRS_toWideband(carrierCtx, numRBs, rnti);

                rankMap(rnti) = dlRank;
                wbCQIMap(rnti) = wbCQI;

                if numel(sbCQI_feat) ~= numSubbandsFeat
                    sbCQI_feat = ones(1, numSubbandsFeat) * wbCQI;
                end
                sbCQIMap(rnti,:) = sbCQI_feat;

                WprgMap{rnti} = Wprg; % [RI x P x 1] WIDEBAND

                % --- features ---
                fR = min(obj.AvgThroughputMbps(rnti) / 100, 1);
                fH = min(dlRank / 2, 1);
                fD = obj.AvgAllocRatio(rnti);
                fB = ueCtx.BufferStatusDL;

                cqiIdxWB = min(max(round(wbCQI), 0), 15);
                fO = obj.CQIToSE(cqiIdxWB + 1) / 5.5547;

                sbIdx = min(max(round(sbCQI_feat), 0), 15);
                fG = obj.CQIToSE(sbIdx + 1) / 5.5547;

                % WIDEBAND: compute single rho value (no PRG dimension)
                if ~isempty(scheduledUEsWideband)
                    fRho = obj.computeRhoWideband(WprgMap, scheduledUEsWideband, rnti);
                else
                    prevUEs = eligibleUEs(1:max(k-1,1));
                    fRho = obj.computeRhoWideband(WprgMap, prevUEs, rnti);
                end

                % WIDEBAND: single rho value, replicate to feature subbands for compatibility
                fRho_feat = fRho * ones(1, numSubbandsFeat);

                exportMatrix(rnti,:) = [fR,fH,fD,fB,fO, fG, fRho_feat];
            end

            % ---- DRL decision: allocationMatrix [numRBGs x MaxNumLayers] ----
            allocationMatrix = obj.communicateWithPythonTTI(exportMatrix, eligibleUEs, numSubbandsFeat, numRBs, numRBGs);

            % ---- Apply WIDEBAND kappa filter ----
            [finalUEs, finalFreqAlloc, finalMCS, finalW] = obj.applyAllocationMatrixWideband( ...
                allocationMatrix, eligibleUEs, WprgMap, sbCQIMap, carrierCtx0, rbgSize, schedulingInfo.MaxNumUsersTTI);

            % ---- update metrics (simple) ----
            servedBytes = zeros(1, obj.MaxUEs);
            currentAlloc = zeros(1, obj.MaxUEs);
            numRBGTotal = numRBGs * obj.MaxNumLayers;

            for i = 1:numel(finalUEs)
                u = finalUEs(i);
                nRBG = sum(finalFreqAlloc(i,:));
                if nRBG > 0
                    bpp = obj.getBytesPerPRB(finalMCS(i));
                    servedBytes(u) = nRBG * rbgSize * bpp;
                    currentAlloc(u) = (nRBG * max(1, obj.getNumLayersFromW(finalW{i}))) / numRBGTotal;
                end
            end

            obj.LastServedBytes = servedBytes;
            obj.LastAllocRatio  = currentAlloc;
            obj.AvgAllocRatio   = obj.RhoEWMA * obj.AvgAllocRatio + (1-obj.RhoEWMA) * currentAlloc;

            instRateMbps = (servedBytes * 8) / 1e6;
            obj.AvgThroughputMbps = obj.RhoEWMA * obj.AvgThroughputMbps + (1-obj.RhoEWMA) * instRateMbps;

            % ---- output grants (IMPORTANT: W must be RI x P x NPRG) ----
            dlAssignments = obj.DLGrantArrayStruct(1:numel(finalUEs));
            
            % Log precoding info for each UE
            fprintf('\n=== PRECODING ASSIGNMENT LOG (TTI) ===\n');
            
            for i = 1:numel(finalUEs)
                u = finalUEs(i);
                dlAssignments(i).RNTI = u;
                dlAssignments(i).GNBCarrierIndex = carrierIndex;
                dlAssignments(i).FrequencyAllocation = finalFreqAlloc(i,:);

                carrierCtx = obj.UEContext(u).ComponentCarrier(carrierIndex);
                mcsOffset = fix(carrierCtx.MCSOffset(obj.DLType+1));
                dlAssignments(i).MCSIndex = min(max(finalMCS(i) - mcsOffset, 0), 27);

                % WIDEBAND: W is RI x P x 1
                dlAssignments(i).W = finalW{i};
                
                % Log precoding details for this UE
                W = finalW{i};
                if ~isempty(W) && ~isscalar(W)
                    numLayers = size(W, 1);
                    numPorts = size(W, 2);
                    fprintf('  UE%2d: Rank=%d, Ports=%d (WIDEBAND Precoding)\n', ...
                        u, numLayers, numPorts);
                    fprintf('        MCS=%d (raw=%d, offset=%d), RBGs=%d\n', ...
                        dlAssignments(i).MCSIndex, finalMCS(i), mcsOffset, sum(finalFreqAlloc(i,:)));
                else
                    fprintf('  UE%2d: W=scalar/empty, MCS=%d\n', u, dlAssignments(i).MCSIndex);
                end
            end
            fprintf('======================================\n\n');
        end
    end

    % =========================
    % ====== CORE (PRG) =======
    % =========================
    methods (Access = protected)

        function rho = computeRhoWideband(obj, WprgMap, scheduledUEs, rnti)
            % WIDEBAND: compute single kappa value (no PRG dimension)
            % rho = max_c kappa(u, c) over all co-scheduled UEs
            rho = 0;

            Wu = WprgMap{rnti};
            if isempty(Wu), return; end

            if iscell(scheduledUEs)
                ueList = scheduledUEs{1};  % wideband: only 1 PRG
            else
                ueList = scheduledUEs;
            end
            if isempty(ueList), return; end
            ueList = ueList(ueList ~= rnti);

            for k = 1:numel(ueList)
                c = ueList(k);
                if c > numel(WprgMap), continue; end
                Wc = WprgMap{c};
                if isempty(Wc), continue; end
                rho = max(rho, obj.kappaWideband(Wu, Wc));
            end
        end

        function kappa = kappaWideband(obj, W1, W2)
            % WIDEBAND correlation: W1/W2 are [RI x P x 1] or [RI x P]
            P1 = obj.getPrecodingWideband(W1);  % [P x RI1]
            P2 = obj.getPrecodingWideband(W2);  % [P x RI2]
            if isempty(P1) || isempty(P2)
                kappa = 0; return;
            end

            % P1, P2 are [NumPorts x NumLayers] after transpose
            % Normalize each column (layer) to unit norm
            P1n = P1 ./ max(vecnorm(P1, 2, 1), 1e-10);
            P2n = P2 ./ max(vecnorm(P2, 2, 1), 1e-10);

            % Correlation matrix: [RI1 x RI2]
            C = P1n' * P2n;
            
            % Max absolute correlation
            kappa = max(abs(C(:)));
        end

        function P = getPrecodingWideband(~, Wprg)
            % Extract wideband precoder: W is [RI x P x 1] or [RI x P]
            % Returns [P x RI] for correlation computation
            if isempty(Wprg), P = []; return; end
            if ndims(Wprg) == 3
                P = Wprg(:,:,1).';  % [RI x P] -> [P x RI]
            else
                P = Wprg.';  % [RI x P] -> [P x RI]
            end
        end

        function X = normalizeColumns(~, X)
            if isempty(X), return; end
            for j = 1:size(X,2)
                n = norm(X(:,j));
                if n > 0
                    X(:,j) = X(:,j) / n;
                end
            end
        end

        % NOTE: compressRhoToFeatureSubbands removed - not needed for wideband

        function [finalUEs, finalFreqAlloc, finalMCS, finalW] = applyAllocationMatrixWideband( ...
                obj, allocationMatrix, eligibleUEs, WprgMap, sbCQIMap, carrierCtx0, rbgSize, maxUsersTTI)
            % WIDEBAND PRECODING version - single PRG for entire bandwidth

            numRBs = obj.CellConfig(1).NumResourceBlocks;

            [numRBGs, ~] = size(allocationMatrix);
            tempFreq = zeros(obj.MaxUEs, numRBGs);
            tempW = cell(obj.MaxUEs, 1);

            % track worst kappa per UE over feature-subband for MCS backoff
            numSubbandsFeat = size(sbCQIMap,2);
            ueWorstRhoFeat = zeros(obj.MaxUEs, numSubbandsFeat);

            for rbg = 1:numRBGs
                rbStart = (rbg-1)*rbgSize + 1;

                % WIDEBAND: no PRG index needed (single PRG spanning all RBs)

                % feature subband index (for CQI/MCS aggregation)
                sbFeat = ceil(rbStart / obj.SubbandSize);
                sbFeat = min(max(sbFeat,1), numSubbandsFeat);

                % 1) collect scheduled UEs on this RBG
                uelist = unique(allocationMatrix(rbg,:));
                uelist = uelist(uelist>0 & ismember(uelist, eligibleUEs));
                if isempty(uelist), continue; end

                % 2) cap users per RBG
                if numel(uelist) > obj.MaxUsersPerRBG
                    uelist = obj.keepBestBySubbandCQI(uelist, sbCQIMap, sbFeat, obj.MaxUsersPerRBG);
                end

                % 3) WIDEBAND kappa-based pruning (single PRG)
                [uelist2, dropped, corrAtDrop] = obj.filterByKappaWideband(uelist, WprgMap);
                if obj.MU_CorrLog && ~isempty(dropped)
                    fprintf('[MU-CORR] RBG %d (Wideband) drop %s (corrAtDrop=%.3f thr=%.3f)\n', ...
                        rbg, mat2str(dropped), corrAtDrop, obj.MU_CorrThreshold);
                end

                % 4) apply frequency alloc + store W (wideband: RI x P x 1)
                for u = uelist2
                    tempFreq(u, rbg) = 1;
                    if isempty(tempW{u}) && u <= numel(WprgMap) && ~isempty(WprgMap{u})
                        tempW{u} = WprgMap{u}; % RI x P x 1 (WIDEBAND)
                    end
                end

                % 5) update worst rho for MCS backoff (wideband: same for all subbands)
                if numel(uelist2) > 1
                    for a = 1:numel(uelist2)
                        u = uelist2(a);
                        maxR = 0;
                        for b = 1:numel(uelist2)
                            if a==b, continue; end
                            v = uelist2(b);
                            if isempty(WprgMap{u}) || isempty(WprgMap{v}), continue; end
                            maxR = max(maxR, obj.kappaWideband(WprgMap{u}, WprgMap{v}));
                        end
                        ueWorstRhoFeat(u, sbFeat) = max(ueWorstRhoFeat(u, sbFeat), maxR);
                    end
                end
            end

            % scheduled UEs
            scheduledUEs = find(sum(tempFreq,2) > 0);

            % limit users per TTI
            if numel(scheduledUEs) > maxUsersTTI
                allocCounts = sum(tempFreq(scheduledUEs,:),2);
                [~,ix] = sort(allocCounts,'descend');
                scheduledUEs = scheduledUEs(ix(1:maxUsersTTI));
            end

            finalUEs = scheduledUEs.';
            finalFreqAlloc = tempFreq(scheduledUEs,:);
            finalW = tempW(scheduledUEs);

            % subband-aware MCS + MU backoff + WIDEBAND backoff
            finalMCS = zeros(numel(finalUEs),1);
            for i = 1:numel(finalUEs)
                u = finalUEs(i);

                sbSet = obj.getAllocatedFeatureSubbandsForUE(finalFreqAlloc(i,:), rbgSize, numSubbandsFeat);
                if isempty(sbSet)
                    cqiEff = 7;
                    rhoEff = 0;
                else
                    % Use MIN CQI instead of mean for more conservative estimate
                    cqiEff = min(sbCQIMap(u, sbSet));
                    rhoEff = max(ueWorstRhoFeat(u, sbSet));
                end

                cqiEff = min(max(round(cqiEff),1),15);
                mcs = getMCSIndex(obj, cqiEff);

                % MU correlation backoff
                bo_mu = obj.mcsBackoffFromRho(rhoEff);
                
                % Wideband precoding backoff (additional conservative margin)
                bo_wb = obj.WidebandMCSBackoff;
                
                % Total backoff
                bo_total = bo_mu + bo_wb;
                
                finalMCS(i) = max(0, min(27, mcs - bo_total));
                
                % Log MCS selection for debugging BLER
                if obj.MU_CorrLog
                    fprintf('  UE%d: CQI_eff=%.1f -> MCS_raw=%d, bo_mu=%d, bo_wb=%d -> MCS_final=%d\n', ...
                        u, cqiEff, mcs, bo_mu, bo_wb, finalMCS(i));
                end
            end
        end

        function [filteredUEs, droppedUEs, corrAtDrop] = filterByKappaWideband(obj, ueList, WprgMap)
            % WIDEBAND kappa-based filtering (no PRG index needed)
            filteredUEs = ueList(:).';
            droppedUEs = [];
            corrAtDrop = 0;

            while numel(filteredUEs) > 1
                worst = -Inf;
                pair = [];

                for i = 1:numel(filteredUEs)-1
                    for j = i+1:numel(filteredUEs)
                        a = filteredUEs(i);
                        b = filteredUEs(j);
                        if a > numel(WprgMap) || b > numel(WprgMap), continue; end
                        if isempty(WprgMap{a}) || isempty(WprgMap{b}), continue; end
                        corr = obj.kappaWideband(WprgMap{a}, WprgMap{b});
                        if corr > worst
                            worst = corr;
                            pair = [a b];
                        end
                    end
                end

                if isempty(pair) || worst <= obj.MU_CorrThreshold
                    break;
                end

                corrAtDrop = max(corrAtDrop, worst);

                % deterministic drop: drop higher UE id
                dropUE = max(pair);
                droppedUEs(end+1) = dropUE; %#ok<AGROW>
                filteredUEs(filteredUEs == dropUE) = [];
            end
        end

        function bo = mcsBackoffFromRho(obj, rho)
            if rho <= obj.MU_MCSBackoffStartRho
                bo = obj.MU_MCSBackoffMin;
                return;
            end
            if rho >= obj.MU_MCSBackoffFullRho
                bo = obj.MU_MCSBackoffMax;
                return;
            end
            t = (rho - obj.MU_MCSBackoffStartRho) / (obj.MU_MCSBackoffFullRho - obj.MU_MCSBackoffStartRho);
            bo = round(obj.MU_MCSBackoffMin + t*(obj.MU_MCSBackoffMax - obj.MU_MCSBackoffMin));
        end

        function sbSet = getAllocatedFeatureSubbandsForUE(obj, ueRbgRow, rbgSize, numSubbandsFeat)
            rbgIdx = find(ueRbgRow > 0);
            sbSet = [];
            for r = rbgIdx(:).'
                rbStart = (r-1)*rbgSize + 1;
                sb = ceil(rbStart / obj.SubbandSize);
                sb = min(max(sb,1),numSubbandsFeat);
                sbSet(end+1) = sb; %#ok<AGROW>
            end
            sbSet = unique(sbSet);
        end

        function uelist = keepBestBySubbandCQI(~, uelist, sbCQIMap, sb, K)
            cqi = sbCQIMap(uelist, sb);
            cqi(isnan(cqi)) = -Inf;
            [~,ix] = sort(cqi,'descend');
            uelist = uelist(ix(1:min(K,numel(uelist))));
        end

        function scheduledUEs = getScheduledUEsWideband(~, allocationMatrix)
            % WIDEBAND: Get all unique scheduled UEs from last allocation
            % No PRG mapping needed - single wideband precoder
            allUEs = allocationMatrix(:);
            scheduledUEs = unique(allUEs(allUEs > 0)).';
        end
    end

    % =========================
    % ===== CSI -> WIDEBAND W =
    % =========================
    methods (Access = protected)

        function [dlRank, wbCQI, sbCQI_feat, Wwideband] = decodeCSIRS_toWideband(obj, carrierCtx, numRBs, rnti)
            % WIDEBAND version - Output:
            % - dlRank
            % - wbCQI
            % - sbCQI_feat: length = ceil(NRB/SubbandSize) (feature subbands for DRL)
            % - Wwideband: RI x P x 1  (single wideband precoder)

            % Prefer cached CSI from updateChannelQualityDL
            csi = obj.getLatestCSIRSReport(rnti, carrierCtx);
            if rnti <= numel(obj.LastCSIRSReport) && ~isempty(obj.LastCSIRSReport{rnti})
                csi = obj.LastCSIRSReport{rnti};
            end

            % Rank
            if isfield(csi,'RI') && ~isempty(csi.RI)
                dlRank = csi.RI;
            else
                dlRank = 1;
            end

            % CQI -> feature subbands
            numSubbandsFeat = ceil(numRBs / obj.SubbandSize);
            if isfield(csi,'CQI') && ~isempty(csi.CQI)
                raw = csi.CQI;
                if isscalar(raw)
                    wbCQI = raw;
                    sbCQI_feat = ones(1,numSubbandsFeat)*raw;
                else
                    wbCQI = mean(raw,'all');
                    tmp = raw(:).';
                    % resize to feature subbands
                    if numel(tmp) ~= numSubbandsFeat
                        tmp = imresize(tmp, [1 numSubbandsFeat], 'nearest');
                    end
                    sbCQI_feat = tmp;
                end
            else
                wbCQI = 0;
                sbCQI_feat = zeros(1,numSubbandsFeat);
            end
            wbCQI = min(max(round(wbCQI),0),15);
            sbCQI_feat = min(max(round(sbCQI_feat),0),15);

            % ---- WIDEBAND PRECODING: single precoder for entire bandwidth ----
            Pports = obj.CellConfig(1).NumTransmitAntennas;

            if ~isfield(csi,'W') || isempty(csi.W)
                % fallback: omnidirectional precoder
                Wwideband = complex(zeros(dlRank, Pports, 1));
                Wwideband(:,:,1) = (ones(Pports, dlRank)./sqrt(Pports)).'; % RI x P
                return;
            end

            Wraw = csi.W;

            if ismatrix(Wraw)
                % Wideband case: likely P x RI or RI x P
                % Want RI x P x 1
                if size(Wraw,1) ~= dlRank && size(Wraw,2) == dlRank
                    W2 = Wraw.'; % RI x P
                else
                    W2 = Wraw;
                    if size(W2,1) ~= dlRank
                        W2 = W2.';
                    end
                end

                Pports = size(W2,2);
                Wwideband = complex(zeros(dlRank, Pports, 1));
                Wwideband(:,:,1) = W2;
                return;
            end

            % 3D case: P x RI x NSB  OR  RI x P x NSB
            % For wideband: average or take first subband
            sz = size(Wraw);

            if sz(2) == dlRank
                % assume P x RI x NSB
                WriP = permute(Wraw, [2 1 3]);  % RI x P x NSB
            elseif sz(1) == dlRank
                % assume RI x P x NSB already
                WriP = Wraw;
            else
                % unknown shape -> fallback omnidirectional
                Wwideband = complex(zeros(dlRank, Pports, 1));
                Wwideband(:,:,1) = (ones(Pports, dlRank)./sqrt(Pports)).';
                return;
            end

            Pports = size(WriP,2);
            
            % WIDEBAND: use average across all subbands (or first subband)
            % Using mean provides better wideband approximation
            Wwideband = complex(zeros(dlRank, Pports, 1));
            Wwideband(:,:,1) = mean(WriP, 3);  % average across subbands
        end

        function csiReport = getLatestCSIRSReport(obj, rnti, carrierCtx)
            csiReport = struct();
            raw = [];
            if ~isempty(carrierCtx) && isfield(carrierCtx,'CSIMeasurementDL') && isfield(carrierCtx.CSIMeasurementDL,'CSIRS')
                raw = carrierCtx.CSIMeasurementDL.CSIRS;
            end
            if isempty(raw) && rnti <= numel(obj.UEContext)
                ueCtx = obj.UEContext(rnti);
                if isfield(ueCtx,'CSIMeasurementDL') && isfield(ueCtx.CSIMeasurementDL,'CSIRS')
                    raw = ueCtx.CSIMeasurementDL.CSIRS;
                end
            end
            if ~isempty(raw)
                csiReport = raw;
            end
        end
    end

    % =========================
    % ====== DRL comm =========
    % =========================
    methods (Access = protected)

        function drlSendJSON(obj, S)
            if ~obj.DRL_IsConnected, error('DRL not connected'); end
            msg = jsonencode(S);
            data = [uint8(msg) obj.DRL_Terminator];
            write(obj.DRL_Socket, data, "uint8");
        end

        function S = drlRecvJSON(obj, timeoutSec)
            if nargin < 2, timeoutSec = obj.DRL_TimeoutSec; end
            t0 = tic;

            while true
                nb = obj.DRL_Socket.BytesAvailable;
                if nb > 0
                    chunk = read(obj.DRL_Socket, nb, "uint8");
                    obj.DRL_RxBuf = [obj.DRL_RxBuf; chunk(:)];
                end

                idx = find(obj.DRL_RxBuf == obj.DRL_Terminator, 1, 'first');
                if ~isempty(idx)
                    line = obj.DRL_RxBuf(1:idx-1);
                    obj.DRL_RxBuf = obj.DRL_RxBuf(idx+1:end);
                    if isempty(line), continue; end
                    S = jsondecode(char(line(:)'));
                    return;
                end

                if toc(t0) > timeoutSec
                    error('Timeout waiting DRL JSON response');
                end
                pause(0.002);
            end
        end

        function allocationMatrix = communicateWithPythonTTI(obj, exportMatrix, eligibleUEs, numSubbandsFeat, numRBs, numRBGs)
            if ~obj.DRL_IsConnected
                error('DRL not connected (call connectToDRLAgent first)');
            end

            payload = struct();
            payload.type = "TTI_OBS";
            payload.frame = double(obj.CurrFrame);
            payload.slot  = double(obj.CurrSlot);
            payload.max_ues = obj.MaxUEs;
            payload.max_layers = obj.MaxNumLayers;
            payload.num_rbg = numRBGs;
            payload.num_subbands = numSubbandsFeat;
            payload.num_rbs = numRBs;
            payload.eligible_ues = eligibleUEs;
            payload.features = exportMatrix;

            obj.drlSendJSON(payload);
            resp = obj.drlRecvJSON(obj.DRL_TimeoutSec);

            if ~isfield(resp,"type") || resp.type ~= "TTI_ALLOC"
                error("Invalid response type");
            end
            allocationMatrix = resp.allocation;
            if ~isequal(size(allocationMatrix), [numRBGs obj.MaxNumLayers])
                error("allocation size mismatch");
            end

            allocationMatrix(~ismember(allocationMatrix, [0 eligibleUEs])) = 0;
            obj.LastAllocationMatrix = allocationMatrix;
        end
    end

    % =========================
    % ===== misc helpers ======
    % =========================
    methods (Access = protected)

        function bpp = getBytesPerPRB(~, mcs)
            effs = [0.15 0.23 0.38 0.60 0.88 1.18 1.48 1.91 2.40 2.73 3.32 3.90 4.52 5.12 5.55 6.07 6.23 6.50 6.70 6.90 7.00 7.10 7.20 7.30 7.35 7.40 7.45 7.48 7.50];
            mcs = min(max(mcs,0),28);
            bpp = (effs(mcs+1) * 12 * 14 * 0.9) / 8;
        end

        function numLayers = getNumLayersFromW(~, W)
            % W is RI x P x NPRG
            if isempty(W), numLayers = 1; return; end
            sz = size(W);
            if numel(sz) >= 2
                numLayers = sz(1);
            else
                numLayers = 1;
            end
        end
    end
end
