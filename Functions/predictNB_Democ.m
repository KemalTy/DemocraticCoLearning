function [predictions, scores] = predictNB_Democ( model, feats )
	 

    [predictions, scores] = predict(model, feats);
    
    % First index in confidences is always the predicted class
    scores = sort(scores, 'descend');

end

