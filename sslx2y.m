function errors = sslx2y(X, Y, method, seed)
    rng(seed);

    switch method
        case 'linear'
            train = @fitlm;
        otherwise
            train = @fitlm;
    end

    % tunable hyperparameters
    conf_thres = 0.1;
    max_update = 500;
    % sample size labelled set =  ls_size_offset + no. features
    ls_size_offset = 5; % could be zero theoretically -- but there would be numerous ill-conditioned warnings.

    % for feature = binary with only 4 features
    % a very small labelled dataset may be all 0 or 1
    % may need to set a bigger value to avoid Rank Deficiency
    % ls_size_offset = 46;


    % input: X, Y, method (to be expanded)
    % function: test if semi-supervised learning (ssl) is effective for predicting Y from X
    % output: [RMSE_without_ssl, RMSE_with_ssl]

    % mdl_base: training with labeled set only
    % mdl_ssl: training with both labeled and unlabeled sets using ssl
    
    % target: supervised labels
    % pred_sup: predictors for supervised learning
    % pred_ssl: predictors for semi-supervised learning

    assert(size(X,1) > size(X,2) + ls_size_offset, 'X is too wide');
    assert(size(X,2) > 1, 'X is too narrow');

    % randomize indices for labeled set and unlabeled set
    ls_indices = randperm(size(X,1));
    ls_indices = ls_indices(1:(size(X,2) + ls_size_offset));
    uls_indices = setdiff(1:size(X,1), ls_indices);

    ls_y = Y(ls_indices);
    ls_x = X(ls_indices,:);

    if (rank(ls_x) < size(ls_x, 2))
        errors = [0,0];
        return
    end

    % mdl_base contains two predictors for co-training
    mdl_base = {train(ls_x(:,2:end), ls_y), train(ls_x(:,1:end-1), ls_y)};

    mdl_ssl = mdl_base;
    uls_x = X(uls_indices,:);

    current_x = ls_x;
    current_y = ls_y;
    
    tsize_current = 0;
    while (tsize_current < size(current_x, 1))
        tsize_current = size(current_x, 1);

        % tempb: use the current ensemble to predict Y
        % conf: use the difference between two predictions as a proxy of
        % predictive confidence
        % take the pseudolabel of unlabeled samples with difference below
        % conf_thres - if more than max_update samples satisfy this, take 
        % max_update samples with the minimum difference only
        
        tempb = [predict(mdl_ssl{1}, uls_x(:,2:end)), predict(mdl_ssl{2}, uls_x(:,1:end-1))];
        conf = abs(tempb(:,2)-tempb(:,1));
        mv_indx = find(conf < min(max(mink(conf,max_update)), conf_thres));

        % update predictors and pseudolabels (as mean of ensemble
        % predictions), remove from unlabeled set
        
        current_x = [current_x; uls_x(mv_indx, :)];
        current_y = [current_y; mean(tempb(mv_indx, :)')'];
        uls_x = uls_x(setdiff(1:size(uls_x, 1), mv_indx),:);

        % update semi-supervised model
        mdl_ssl = {train(current_x(:,2:end), current_y), ...
            train(current_x(:,1:end-1), current_y)};
    end
    
    test_x = X(uls_indices,:);
    test_y = Y(uls_indices);
    
    % base_res: final result of supervised learning
    % semi_res: final result of semi-supervised learning
    
    result_base = fitlm([predict(mdl_base{1}, test_x(:,2:end)), predict(mdl_base{2}, test_x(:,1:end-1))], test_y, 'Linear');
    result_ssl = fitlm([predict(mdl_ssl{1}, test_x(:,2:end)), predict(mdl_ssl{2}, test_x(:,1:end-1))], test_y, 'Linear');

    errors = [result_base.RMSE, result_ssl.RMSE];
end