function [Decode_WT] = dec_boot(cell_type, mouse, train_id, test_id, num_cells, perm, seed)
    rng(seed); % fix the random number seed for reproducibility

    Decode_WT.Binary.MCReps = 1; % Number of Monte-carlo shufflings 5
    Decode_WT.Binary.nPerms = 1; % Number of permutation for statistical significance 25
    opts.Regularization = 'ridge';
    opts.ClassNames = [-1 1];
    opts.Prior = 'empirical'; %prior probability
    opts.ObservationsIn = 'columns';
    opts.Learner = 'svm';

    Decode_WT.Binary.Real = struct();
    Decode_WT.Binary.Perm = struct();

    session_ids = [train_id, test_id];

    for i = 1:length(session_ids)
        session_id = session_ids(i);
        dir1 = sprintf('session%d', session_id);
        dir2 = sprintf('%s_session%d', mouse, session_id);
        p = fileparts(mfilename('fullpath'));
        root = fullfile(p, '..', '..', '..', 'Data', 'Decoding', cell_type, mouse, dir1);
        f_dm = fullfile(root, dir2);

        switch i
            case 1
                train_dm = load(f_dm);
            case 2
                test_dm = load(f_dm);
        end
    end

    train_raster = (train_dm.raster)';
    train_trial = train_dm.trial;
    train_stimulus = (train_dm.stimulus)';
    train_trial_length = length(unique(train_trial));
    train_group = zeros(train_trial_length, 1);

    test_raster = (test_dm.raster)';
    test_trial = test_dm.trial;
    test_stimulus = (test_dm.stimulus)';
    test_trial_length = length(unique(test_trial));
    test_group = zeros(test_trial_length, 1);

    total_cells = size(train_raster,1);
    % Select random cells for training data
    Decode_WT.Binary.cell_id = datasample(1:total_cells, num_cells, 'Replace', false); % select indices randomly
    % Subset the training data with the selected indices
    train_raster = train_raster(Decode_WT.Binary.cell_id, :);
    test_raster = test_raster(Decode_WT.Binary.cell_id, :);

    for ti = 1:train_trial_length
        stimval = train_stimulus(train_trial == ti);
        train_group(ti) = stimval(1);
    end

    for ti = 1:test_trial_length
        stimval = test_stimulus(test_trial == ti);
        test_group(ti) = stimval(1);
    end

    if train_id == test_id
        cvp = cvpartition(train_group, 'KFold', 5, 'Stratify', true);
        train_cvp = cvp;
        test_cvp = cvp;
        train_test = 1;
    else
        train_cvp = cvpartition(train_group, 'KFold', 5, "Stratify", true);
        test_cvp = cvpartition(test_group, 'KFold', 5, "Stratify", true);
        train_test = 0;
    end

    % 5-fold cross-validation and prediction
    [Decode_WT.Binary.Real.label, ...
        Decode_WT.Binary.Real.confmat, ...
        Decode_WT.Binary.Real.accuracy,...
        Decode_WT.Binary.Real.weight] = ...
        SVM(@fitclinear, train_raster, train_stimulus, ...
        train_trial, test_raster, test_stimulus, test_trial, ...
        train_cvp, test_cvp, opts, train_test, Decode_WT.Binary.MCReps, 'real');

    % Permutation test(null model)
    [Decode_WT.Binary.Perm.label, ...
        Decode_WT.Binary.Perm.confmat, ...
        Decode_WT.Binary.Perm.accuracy,...
        Decode_WT.Binary.Perm.weight] = ...
        SVM(@fitclinear, train_raster, train_stimulus, ...
        train_trial, test_raster, test_stimulus, test_trial, ...
        train_cvp, test_cvp, opts, train_test, Decode_WT.Binary.nPerms, 'randperm');
end

%% Functions
function [label, confmat, accuracy, weights] = SVM(func, train_raster, train_stimulus, train_trial, ...
        test_raster, test_stimulus, test_trial, ...
        train_cvp, test_cvp, opts, train_test, nReps, method)

    nClasses = numel(opts.ClassNames);
    label = zeros(numel(test_stimulus), nReps); %class labels
    confmat = zeros(nClasses, nClasses, nReps); %confusion matrix
    accuracy = zeros(nReps, 1);
    num_folds = train_cvp.NumTestSets; % number of cross-vallidation fold
    weights_fold = zeros(size(train_raster,1),num_folds, nReps);

    cellopts = [fieldnames(opts) struct2cell(opts)]';

    timerVal = tic; % for displaying elapsed time

    for ii = 1:nReps
        if mod(ii, 10) == 0
            fprintf('Processing %d/%d iterations...elapsed %.3f seconds.\n', ...
                ii, nReps, toc(timerVal));
        end

        if train_test == 1 % same session decoding cross validation fold
            cvp = repartition(train_cvp);
            train_cvp_rep = cvp;
            test_cvp_rep = cvp;
        else % across session decoding cross validation fold
            train_cvp_rep = repartition(train_cvp);
            test_cvp_rep = repartition(test_cvp);
        end

        label_predicted = {};
        true_label = {};

        switch method
            case 'randperm' % Shuffle labels in 1-second trial units without completely disrupting the temporal structure
                shuffled_Idx = randperm(length(unique(train_trial)));

                for i = 1:length(shuffled_Idx)
                    idx = shuffled_Idx(i);
                    mask = train_trial == idx;
                    shuffled_stimulus{i} = train_stimulus(:, mask);
                end

                train_stimulus = cell2mat(shuffled_stimulus);

            case 'circshift'
                shuffled_Idx = circshift(length(unique(train_trial)));
        end

        for fold = 1:train_cvp_rep.NumTestSets
            trainIdx = training(train_cvp_rep, fold);
            testIdx = test(test_cvp_rep, fold);
            trainID = find(trainIdx);
            testID = find(testIdx);
            N_train = {};
            C_train = {};
            N_test = {};
            C_test = {};

            for i = 1:length(trainID)
                idx = trainID(i);
                N_train{i} = train_raster(:, train_trial == idx);
                C_train{i} = train_stimulus(train_trial == idx);
            end

            for i = 1:length(testID)
                idx = testID(i);
                N_test{i} = test_raster(:, test_trial == idx);
                C_test{i} = test_stimulus(test_trial == idx);
            end

            N_train = cell2mat(N_train);
            C_train = cell2mat(C_train);
            N_test = (cell2mat(N_test))';
            C_test = (cell2mat(C_test))';

            % weight adjustment
            wt = ones(length(C_train), 1);
            c = unique(C_train);
            for ci = 1:2
                pick = C_train == c(ci);
                if sum(pick) > 0
                    wt(pick) = 1 ./ sum(pick);
                end
            end

            Mdl = func(N_train, C_train, 'Weights', wt, cellopts{:}); %train model

            %Store the weights for each fold
            weights_fold(:, fold, ii) = Mdl.Beta;
            %test model
            [label_predicted{fold, 1}, ~] = predict(Mdl, N_test);
            true_label{fold, 1} = C_test;
        end
        label_predicted = cell2mat(label_predicted);
        true_label = cell2mat(true_label);
        weights.mean(:,ii) = mean(weights_fold,2);
        weights.std(:,ii) = std(weights_fold,[],2);

        confmat(:, :, ii) = confusionmat(true_label, label_predicted);
        accuracy(ii) = trace(confmat(:, :, ii)) / numel(true_label);
    end
end