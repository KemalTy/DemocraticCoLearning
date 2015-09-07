function [ predictions, confidences ] = predictSVM_Democ( model, feats )
 

% Do predictions on data
dummyLabels = ones( size(feats,1), 1);

if(isunix) % OS independent
	[predictions, ~, confidences] = svmpredict(dummyLabels, feats, model, '-q -b 1');
else
	[predictions, ~, confidences] = libsvmpredict(dummyLabels, feats, model, '-q -b 1');
end

% First index in confidences is always the predicted class
confidences = sort(confidences, 'descend');

end

