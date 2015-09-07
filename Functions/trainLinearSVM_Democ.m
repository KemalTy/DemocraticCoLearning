% Trains svm on data
function [ model ] = trainLinearSVM_Democ( labels, feats )
 fprintf('Training Log. Reg...\n');

%% Options
type = '0'; 

%% Unlabeled data must be invisible to SVM
unlabIdx = find( labels == 0 );
feats(unlabIdx, :) = [];
labels(unlabIdx) = [];

%% Run SVM
% I will have:
% * 5-fold cross validation
% * Parameter selection
% * Weighted misclassification cost w.r.t. class instance number (b/c
% of class imbalance)

%% Set misclassification weights
numPos	= length( find(labels > 0) );
numNeg = length( find(labels == -1) );
ratio = numNeg / numPos;	% I assume that |-| > |+|

wNeg = 1;
wPos  = round( ratio );

fprintf('-/+: %f...\n', ratio);
fprintf('Weight +: %d...\n', wPos);
fprintf('Weight -: %d...\n', wNeg);

% Find best parameters with 5 fold CV
cmd = ['-q -C -v 5', ' -s ', num2str(type), ' -w-1 ' num2str(wNeg), ' -w1 ' num2str(wPos)];
cvinfo = train(labels,sparse(feats), cmd);
bestc = cvinfo(1); bestcv = cvinfo(2);
fprintf('Best C: %d, BestCV: %d...\n', bestc, bestcv);

% Train with SVM
cmd = [ ' -q', ' -s ', num2str(type), ' -c ', num2str(bestc),' -w-1 ' num2str(wNeg), ' -w1 ' num2str(wPos)];
model = train(labels, sparse(feats), cmd);


end

