% Trains svm on data
function [ model ] = trainSVM_Democ( labels, feats )

 

%% Options
kernel = '0'; % 0 : Linear, 2: RBF
v = '5'; % CV

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
bestcv = 0;
bestc = 0;
bestg = 0;

if(str2num(kernel) == 0)
	for log2c = 5:10,
		cmd = ['-v ' num2str(v) ' -q -c ', num2str(2^log2c),' -t ' num2str(kernel) ' -w-1 ' num2str(wNeg), ' -w1 ' num2str(wPos)];
		
		if(isunix) % OS independent
			cv = svmtrain(labels, feats, cmd);
		else
			cv = libsvmtrain(labels, feats, cmd);
		end
		
		if (cv >= bestcv),
			bestcv = cv; bestc = 2^log2c;
		end
	end
	
elseif(str2num(kernel) == 2)
	for log2c = 5:10,
		for log2g = -10:0
			cmd = ['-v ' num2str(v) ' -q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -t ' num2str(kernel) ' -w-1 ' num2str(wNeg), ' -w1 ' num2str(wPos)];

			if(isunix) % OS independent
				cv = svmtrain(labels, feats, cmd);
			else
				cv = libsvmtrain(labels, feats, cmd);
			end

			if (cv >= bestcv),
				bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
			end
		end
	end
end


% Train with SVM
fprintf('Best CV: %f...\nBest gamma: %f...\nBest C: %f...\n', bestcv, bestg, bestc);
cmdBest = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg),' -t ' num2str(kernel) ' -w-1 ' num2str(wNeg), ' -w1 ' num2str(wPos) ' -b 1'];

if(isunix) % OS independent
    model = svmtrain(labels, feats, cmdBest);
else
    model = libsvmtrain(labels, feats, cmdBest);
end




end

