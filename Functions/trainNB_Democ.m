% Trains Naive Bayes on data
function [ model ] = trainNB_Democ( labels, feats )

	 

    % Remove unlabeled instances
    unlabIdx = find(labels == 0);
    labels(unlabIdx) = [];
    feats(unlabIdx,:) = [];
    
    % Pseudo-train N.N.
    model = fitcnb(feats, labels);

end

