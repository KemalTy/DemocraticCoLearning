function [ predictions, confidences ] = predictLinearSVM_Democ( model, feats )
 

% Do predictions on data
dummyLabels = ones( size(feats,1), 1);
[predictions, ~, confidences] = predict(dummyLabels, sparse(feats), model, '-q -b 1');


% First index in confidences is always the predicted class
confidences = sort(confidences, 'descend');

end

