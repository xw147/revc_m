%% Semi-Supervised Learning Test with Binomial Testing
% This script implements the reverse causality testing method from:
% "Testing of Reverse Causality Using Semi-Supervised Machine Learning"
% by Zhang, Xu, Vaulont, & Zhang (Psychometrika, 2025)
%
% It performs 1,000 iterations (resamples) to test whether semi-supervised
% learning outperforms supervised learning, using a binomial test for
% statistical significance.

clear all;
close all;

%% Configuration
% Number of resamples for binomial test (paper recommends 1,000)
n_resamples = 20000;

% Method: 'linear' for OLS regression
method = 'linear';

% IMPORTANT: Set a fixed seed for the overall experiment reproducibility
% but we'll generate different random splits within each iteration
rng(42, 'twister');

% Variable names for reporting
% y_var_name = 'wfca_w6';
% x_var_names = {'jsm_w7', 'jsm_w8', 'jsm_w9'};


y_var_name = 'jsm_w6';
x_var_names = {'wfca_w7', 'wfca_w8', 'wfca_w9'};

%% Load and prepare data
fprintf('========================================\n');
fprintf('Reverse Causality Test using SSL\n');
fprintf('========================================\n\n');

data = readtable('JS_WFC_4waves_wide.csv');

% Extract variables
Y = data.(y_var_name);
X = table2array(data(:, x_var_names));

% Handle missing values by replacing with zero
fprintf('Handling missing values...\n');
n_missing_Y = sum(isnan(Y));
n_missing_X = sum(isnan(X(:)));

Y(isnan(Y)) = 0;
X(isnan(X)) = 0;

fprintf('  Missing values in Y replaced with 0: %d\n', n_missing_Y);
fprintf('  Missing values in X replaced with 0: %d\n\n', n_missing_X);

fprintf('Variables:\n');
fprintf('  Target (Y): %s\n', y_var_name);
fprintf('  Predictors (X): %s\n', strjoin(x_var_names, ', '));
fprintf('  Sample size: %d observations\n', size(X,1));
fprintf('  Number of predictors: %d\n\n', size(X,2));

%% Run binomial test
fprintf('Running %d resamples for binomial test...\n', n_resamples);
fprintf('(This may take a few minutes)\n\n');

% Initialize counters
ssl_wins = 0;      % SSL outperforms supervised
supervised_wins = 0; % Supervised outperforms SSL
ties = 0;          % Equal performance

% Store all RMSE values for analysis
all_rmse_supervised = zeros(n_resamples, 1);
all_rmse_ssl = zeros(n_resamples, 1);

% Progress tracking
tic;
fprintf('Progress: ');

for iter = 1:n_resamples
    % CRITICAL: Use the SAME seed for each iteration to ensure reproducibility
    % but allow sslx2y to create its own random split
    % This way each iteration gets a truly different random labeled/unlabeled split
    seed = randi(1e9);  % Generate a random seed for this iteration
    
    % Run SSL test
    errors = sslx2y(X, Y, method, seed);
    
    % Store results
    all_rmse_supervised(iter) = errors(1);
    all_rmse_ssl(iter) = errors(2);
    
    % Count wins
    if errors(2) < errors(1)
        ssl_wins = ssl_wins + 1;
    elseif errors(2) > errors(1)
        supervised_wins = supervised_wins + 1;
    else
        ties = ties + 1;
    end
    
    % Progress indicator (every 100 iterations)
    if mod(iter, 100) == 0
        fprintf('%d ', iter);
    end
end

elapsed_time = toc;
fprintf('\nCompleted in %.2f seconds.\n\n', elapsed_time);


%% Calculate binomial test p-values
% Test 1: Excluding ties (per paper's standard approach)
n_excluding_ties = ssl_wins + supervised_wins;
if n_excluding_ties > 0
    % One-tailed binomial test: H0: π = 0.5, H1: π > 0.5
    p_value_excluding_ties = 1 - binocdf(ssl_wins - 1, n_excluding_ties, 0.5);
else
    p_value_excluding_ties = NaN;
end

% Test 2: Including ties (conservative approach, ties count against SSL)
n_including_ties = n_resamples;
% For including ties, we test if SSL wins > 50% of ALL trials
p_value_including_ties = 1 - binocdf(ssl_wins - 1, n_including_ties, 0.5);

%% Display Results
fprintf('========================================\n');
fprintf('RESULTS\n');
fprintf('========================================\n\n');

fprintf('1. Variables Used:\n');
fprintf('   Target (Y): %s\n', y_var_name);
fprintf('   Predictors (X): %s\n\n', strjoin(x_var_names, ', '));

fprintf('2. Resampling Summary:\n');
fprintf('   Total resamples: %d\n', n_resamples);
fprintf('   SSL wins: %d (%.1f%%)\n', ssl_wins, ssl_wins/n_resamples*100);
fprintf('   Supervised wins: %d (%.1f%%)\n', supervised_wins, supervised_wins/n_resamples*100);
fprintf('   Ties: %d (%.1f%%)\n\n', ties, ties/n_resamples*100);

fprintf('3. Binomial Test Results:\n');
fprintf('   -------------------------\n');
fprintf('   p-value (EXCLUDING ties, standards): %.6f\n', p_value_excluding_ties);
fprintf('   p-value (INCLUDING ties): %.6f\n\n', p_value_including_ties);

fprintf('4. RMSE Statistics:\n');
fprintf('   Average RMSE (Supervised):      %.6f (SD: %.6f)\n', ...
    mean(all_rmse_supervised), std(all_rmse_supervised));
fprintf('   Average RMSE (Semi-Supervised): %.6f (SD: %.6f)\n', ...
    mean(all_rmse_ssl), std(all_rmse_ssl));
fprintf('   Average improvement: %.2f%%\n\n', ...
    mean((all_rmse_supervised - all_rmse_ssl) ./ all_rmse_supervised * 100));

%% Interpretation
fprintf('========================================\n');
fprintf('INTERPRETATION (α = 0.05)\n');
fprintf('========================================\n\n');

if p_value_excluding_ties < 0.05
    fprintf('✓ SIGNIFICANT RESULT (excluding ties)\n');
    fprintf('  Semi-supervised learning significantly outperforms supervised learning.\n');
    
else
    fprintf('✗ NOT SIGNIFICANT (excluding ties)\n');
    fprintf('  No statistical evidence that SSL outperforms supervised learning.\n');
   
end

if p_value_including_ties < 0.05
    fprintf('✓ SIGNIFICANT RESULT (including ties)\n');
else
    fprintf('✗ NOT SIGNIFICANT (including ties)\n');
end


fprintf('========================================\n');
