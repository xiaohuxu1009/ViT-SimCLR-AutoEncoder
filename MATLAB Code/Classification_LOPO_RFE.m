clc; clear; close all;

%% Setup the paths
main_folder = 'FOLDER NAME'; % Put folder name here

subfolders = dir(main_folder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

all_true_labels  = [];
all_pred_scores  = [];
X_all            = {};
Y_all            = {};
AUC_all          = [];
accuracy_all     = [];
precision_all    = [];
recall_all       = [];
f1_all           = [];
conf_mat_all     = [];
specificity_all  = [];
feature_importance_all = [];

%% Train
for i = 1:length(subfolders)
    subfolder_path = fullfile(main_folder, subfolders(i).name);
    files = dir(fullfile(subfolder_path, 'feature*.xlsx'));

    for j = 1:length(files)
        filename = fullfile(subfolder_path, files(j).name);
        disp(['Processing: ', filename]);

        A = readmatrix(filename, 'Sheet', 'Train', 'Range', 'train_range');
        B = readmatrix(filename, 'Sheet', 'Val', 'Range', 'val_range');
        C = readmatrix(filename, 'Sheet', 'Test', 'Range', 'test_range');

        labels_A = [zeros(810,1); ones(810,1)];
        labels_B = [zeros(90,1);  ones(90,1)];
        labels_C = [zeros(90,1);  ones(90,1)];

        % 标准化
        [A_zscore, mu, sigma] = zscore(A);
        B_zscore = (B - mu) ./ sigma;
        C_zscore = (C - mu) ./ sigma;

        score_A = A_zscore;
        score_B = B_zscore;
        score_C = C_zscore;

        [coeff, score_A] = pca(score_A, 'NumComponents', 10);
        score_B = score_B * coeff;
        score_C = score_C * coeff;

        % ====== RFE ======
        p         = size(score_A, 2);
        targetNum = max(10, round(p * 0.3));
        stepDrop  = max(1,  round(p * 0.1));

        rfe_res = rfe_svm_linear(score_A, labels_A, ...
            'TargetNum', targetNum, ...
            'Step',      stepDrop,  ...
            'CVFolds',   5);

        sel     = rfe_res.BestFeatureIdx;
        score_A = score_A(:, sel);
        score_B = score_B(:, sel);
        score_C = score_C(:, sel);
        % ==========================================================

        % SVM
        vars = [
            optimizableVariable('BoxConstraint', [1e-1, 1e1], 'Transform', 'log');
            optimizableVariable('KernelScale',   [1e-1, 1e1], 'Transform', 'log')
        ];

        results = bayesopt( ...
            @(params) svmValLoss(params, score_A, labels_A, score_B, labels_B), ...
            vars, ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'Verbose',                 0, ...
            'MaxObjectiveEvaluations', 11, ...
            'PlotFcn',                 []);

        bestParams = results.XAtMinObjective;

        classifier_model = fitcsvm(score_A, labels_A, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint',  bestParams.BoxConstraint, ...
            'KernelScale',    bestParams.KernelScale, ...
            'Standardize',    false);

        % Logic Regression
        % vars = [
        %     optimizableVariable('Lambda', [1e-5, 10], 'Transform', 'log');
        %     optimizableVariable('Regularization', {'ridge', 'lasso'}, 'Type', 'categorical')
        % ];
        % 
        % results = bayesopt(@(params) logRegValLoss(params, score_A, labels_A, score_B, labels_B), ...
        %     vars, ...
        %     'AcquisitionFunctionName', 'expected-improvement-plus', ...
        %     'Verbose', 0, ...
        %     'MaxObjectiveEvaluations', 10, ...
        %     'PlotFcn', []);
        % 
        % bestParams = results.XAtMinObjective;
        % classifier_model = fitclinear(score_A, labels_A, ...
        %     'Learner', 'logistic', ...
        %     'Lambda', bestParams.Lambda, ...
        %     'Regularization', char(bestParams.Regularization), ...
        %     'Solver', 'lbfgs', ...
        %     'GradientTolerance', 1e-6, ...
        %     'BetaTolerance', 1e-6);

        % % Random Forest
        % vars = [
        %     optimizableVariable('NumLearningCycles', [30, 200], 'Type', 'integer');
        %     optimizableVariable('MinLeafSize', [1, 50], 'Type', 'integer');
        %     optimizableVariable('MaxNumSplits', [10, 500], 'Type', 'integer');
        % ];
        % 
        % results = bayesopt(@(params) rfValLoss(params, score_A, labels_A, score_B, labels_B), ...
        %     vars, ...
        %     'AcquisitionFunctionName', 'expected-improvement-plus', ...
        %     'Verbose', 0, ...
        %     'MaxObjectiveEvaluations', 10, ...
        %     'PlotFcn', []);
        % 
        % bestParams = results.XAtMinObjective;
        % classifier_model = fitcensemble(score_A, labels_A, ...
        %     'Method', 'Bag', ...
        %     'NumLearningCycles', bestParams.NumLearningCycles, ...
        %     'Learners', templateTree( ...
        %         'MinLeafSize', bestParams.MinLeafSize, ...
        %         'MaxNumSplits', bestParams.MaxNumSplits));

        beta_sub               = classifier_model.Beta(:);
        importance_full        = zeros(1, size(A_zscore, 2));
        importance_full(sel)   = abs(beta_sub);
        feature_importance_all = [feature_importance_all; importance_full];

        % Prediction
        [~, scores] = predict(classifier_model, score_C);
        predicted_labels_C  = double(scores(:,2) >= 0.5);

        correct_predictions = sum(predicted_labels_C == labels_C);
        accuracy            = correct_predictions / numel(labels_C);
        accuracy_all        = [accuracy_all; accuracy];

        unique_classes = unique(labels_C);
        for c = 1:numel(unique_classes)
            cls       = unique_classes(c);
            idx       = (labels_C == cls);
            class_acc = sum(predicted_labels_C(idx) == labels_C(idx)) / sum(idx);
        end

        % ROC / AUC
        [X, Y, ~, AUC_val] = perfcurve(labels_C, scores(:,2), 1);
        AUC_all      = [AUC_all;   AUC_val];
        X_all{end+1} = X;
        Y_all{end+1} = Y;

        all_true_labels = [all_true_labels; labels_C];
        all_pred_scores = [all_pred_scores; scores(:,2)];

        conf_mat     = confusionmat(double(labels_C), double(predicted_labels_C));
        conf_mat_all = [conf_mat_all; conf_mat];

        TP = conf_mat(2,2);  FN = conf_mat(2,1);
        FP = conf_mat(1,2);  TN = conf_mat(1,1);

        recall      = TP / (TP + FN);
        precision   = TP / (TP + FP);
        f1_score    = 2 * (precision * recall) / (precision + recall);
        specificity = TN / (TN + FP);

        if isnan(precision),   precision   = 0; end
        if isnan(f1_score),    f1_score    = 0; end
        if isnan(specificity), specificity = 0; end

        recall_all      = [recall_all;      recall];
        precision_all   = [precision_all;   precision];
        f1_all          = [f1_all;          f1_score];
        specificity_all = [specificity_all; specificity];
    end
end

%% Plot ROC
max_len  = max(cellfun(@length, X_all));
X_interp = cellfun(@(x) interp1(linspace(0,1,length(x)), x, linspace(0,1,max_len), 'linear'), X_all, 'UniformOutput', false);
Y_interp = cellfun(@(y) interp1(linspace(0,1,length(y)), y, linspace(0,1,max_len), 'linear'), Y_all, 'UniformOutput', false);

X_ALL   = cell2mat(cellfun(@(x) x(:), X_interp, 'UniformOutput', false));
Y_ALL   = cell2mat(cellfun(@(y) y(:), Y_interp, 'UniformOutput', false));

n_folds = size(Y_ALL, 2);
X_mean = mean(X_ALL, 2);
Y_mean = mean(Y_ALL, 2);
Y_STD = std(Y_ALL, 1, 2);   
Y_lo = max(Y_mean - Y_STD, 0);
Y_hi = min(Y_mean + Y_STD, 1);

figure; hold on;
plot(X_mean, Y_mean, 'Color', [0 0.4470 0.7410], 'LineWidth', 2);
fill([X_mean; flipud(X_mean)], ...
     [Y_lo; flipud(Y_hi)], ...
     [0.6 0.6 0.6], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot([0 1], [0 1], 'k--');
xlim([0 1]); ylim([0 1]);
yticks(0:0.2:1);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Mean ROC'));
legend({'Mean ROC', 'SD Band', 'Chance'}, 'Location', 'southeast');

%% Print the Results
fprintf('\n========== Results ==========\n');

n_boot = 2000;
rng(45);

disp(['Accuracy: ', bootstrap_stats(accuracy_all, n_boot)]);
disp(['Specificity: ', bootstrap_stats(specificity_all, n_boot)]);
disp(['Precision: ', bootstrap_stats(precision_all, n_boot)]);
disp(['Recall: ', bootstrap_stats(recall_all, n_boot)]);
disp(['F1-score: ', bootstrap_stats(f1_all, n_boot)]);
disp(['AUC: ', bootstrap_stats(AUC_all, n_boot)]);

fprintf('==============================\n\n');

%% Feature Importance
mean_feature_importance = mean(feature_importance_all, 1);
disp('Feature Importance:');
disp(mean_feature_importance);

figure;
bar(abs(mean_feature_importance), 'FaceAlpha', 0.8);
xlim([0 numel(mean_feature_importance)+1]);
xlabel('Feature Index');
ylabel('Average Importance');
title('Feature Importance');
grid on;

%% ======================== Functions ========================
% SVM validation set loss (for Bayesian optimization)
function errsvm = svmValLoss(params, XTrain, YTrain, XVal, YVal)
    model  = fitcsvm(XTrain, YTrain, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint',  params.BoxConstraint, ...
        'KernelScale',    params.KernelScale, ...
        'Standardize',    false);
    errsvm = loss(model, XVal, YVal);
end

% LR validation set loss (for Bayesian optimization)
function errlr = logRegValLoss(params, XTrain, YTrain, XVal, YVal)
    try
        model = fitclinear(XTrain, YTrain, ...
            'Learner',           'logistic', ...
            'Lambda',            params.Lambda, ...
            'Regularization',    char(params.Regularization), ...
            'Solver',            'lbfgs', ...
            'GradientTolerance', 1e-6, ...
            'BetaTolerance',     1e-6);
        L = loss(model, XVal, YVal);
        if isnan(L), errlr = inf; else, errlr = L; end
    catch
        errlr = inf;
    end
end

% RF validation set loss (for Bayesian optimization)
function errrf = rfValLoss(params, XTrain, YTrain, XVal, YVal)
    try
        t     = templateTree('MinLeafSize',  params.MinLeafSize, ...
                             'MaxNumSplits', params.MaxNumSplits);
        model = fitcensemble(XTrain, YTrain, ...
            'Method',            'Bag', ...
            'NumLearningCycles', params.NumLearningCycles, ...
            'Learners',          t);
        errrf = loss(model, XVal, YVal);
    catch
        errrf = inf;
    end
end

% RFE（For SVM）
function out = rfe_svm_linear(X, y, varargin)
    p  = size(X, 2);
    ip = inputParser;
    addParameter(ip, 'TargetNum',    max(1, round(p/10)),  @(v) isnumeric(v) && isscalar(v));
    addParameter(ip, 'Step',         max(1, round(p/20)),  @(v) isnumeric(v) && isscalar(v));
    addParameter(ip, 'CVFolds',      5,                    @(v) isnumeric(v) && isscalar(v));
    addParameter(ip, 'ClassWeights', []);
    parse(ip, varargin{:});

    kTarget = ip.Results.TargetNum;
    step    = ip.Results.Step;
    K       = ip.Results.CVFolds;
    classW  = ip.Results.ClassWeights;

    if kTarget >= p
        warning('TargetNum >= p, nothing to eliminate.');
    end
    if ~iscategorical(y), y = categorical(y); end

    feat_idx   = 1:p;
    elim_order = [];
    hist_sets  = {};
    hist_loss  = [];

    function mcr = kfold_mcr(Xin, yin, args_local, W)
        c    = cvpartition(yin, 'KFold', K);
        errs = zeros(K, 1);
        for k = 1:K
            tr = training(c, k);
            te = test(c, k);
            if isempty(W)
                mdl_k = fitclinear(Xin(tr,:), yin(tr), args_local{:});
            else
                mdl_k = fitclinear(Xin(tr,:), yin(tr), args_local{:}, 'Weights', W(tr));
            end
            yhat    = predict(mdl_k, Xin(te,:));
            errs(k) = mean(yhat ~= yin(te));
        end
        mcr = mean(errs);
    end

    while numel(feat_idx) > kTarget
        Xsub = X(:, feat_idx);
        args = {'Learner','svm','Regularization','ridge'};

        if isempty(classW)
            W   = [];
            mdl = fitclinear(Xsub, y, args{:});
        else
            W   = resolve_weights(classW, y);
            mdl = fitclinear(Xsub, y, args{:}, 'Weights', W);
        end

        B   = mdl.Beta; if isvector(B), B = B(:); end
        imp = sqrt(sum(B.^2, 2));

        L                = kfold_mcr(Xsub, y, args, W);
        hist_loss(end+1) = L;
        hist_sets{end+1} = feat_idx;

        [~, ord]   = sort(imp, 'ascend');
        drop_now   = ord(1:min(step, numel(feat_idx) - kTarget));
        elim_order = [feat_idx(drop_now), elim_order];
        feat_idx(drop_now) = [];
    end

    Xsub = X(:, feat_idx);
    args = {'Learner','svm','Regularization','ridge'};
    if isempty(classW)
        W         = [];
        mdl_final = fitclinear(Xsub, y, args{:});
    else
        W         = resolve_weights(classW, y);
        mdl_final = fitclinear(Xsub, y, args{:}, 'Weights', W);
    end
    L_final          = kfold_mcr(Xsub, y, args, W);
    hist_loss(end+1) = L_final;
    hist_sets{end+1} = feat_idx;

    [bestLoss, bestIdx] = min(hist_loss);
    bestSet             = hist_sets{bestIdx};

    out.BestFeatureIdx   = bestSet;
    out.BestLoss         = bestLoss;
    out.LossHistory      = hist_loss;
    out.FeatureSets      = hist_sets;
    out.EliminationOrder = elim_order;
    out.FinalModel       = mdl_final;

    function W = resolve_weights(classW, y)
        if isa(classW, 'containers.Map')
            W = arrayfun(@(lbl) classW(lbl), y);
        else
            W = classW;
        end
    end
end

% Indicators
function s = bootstrap_stats(v, ~)
    v          = v(:);
    MAD = mean(abs(v - mean(v)));
    s = sprintf('Mean=%.4f (STD=%.4f) (MAD=%.4f)', ...
        mean(v), std(v, 1), MAD);
end