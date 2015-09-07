% Trains Decision Tree
function [ model ] = trainDTree_Democ( labels, feats )
	 fprintf('Training DTree...\n');

    % Just extract unlabeled instances and return them as model
    unlabIdx = find(labels == 0);
    labels(unlabIdx) = [];
    feats(unlabIdx,:) = [];
    
    model = fitctree(feats, labels);

end

