function [ predictions, confidences ] = predictDTree_Democ( model, feats )

% Predict
[predictions, confidences] = predict(model, feats);

% First index in confidences is always the predicted class
confidences = sort(confidences, 'descend');

end

