function [predictions, scores] = predictNN_Democ_test( model, featsTest)
	
	% Remove unlabeled instances
	labels = model.labels;
	feats = model.feats;
	
    unlabIdx = find(labels == 0);
    labels(unlabIdx) = [];
    feats(unlabIdx,:) = [];
    
    model = fitcknn(feats, labels, 'NumNeighbors', 5);
	[predictions, scores] = predict(model, featsTest);
	
	scores = sort(scores, 'descend');
	
end









