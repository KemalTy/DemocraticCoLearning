function [predictions, scores] = predictNN_Democ( model, f, unlab, isTraining )
	 
	if(isTraining)
		[predictions, scores] = predictUsingGPU2(model, unlab, f);
	else
		[predictions, scores] = predictForTest(model, f);
	end
    
end

function [predictions, scores] = predictUsingMatlab(model, feats)
	[predictions, scores] = predict(model, feats);
end

function [predictions, scores] = predictForTest(model, featsTest)
	
	% Remove unlabeled instances
	labels = model.labels;
	feats = model.feats;
	
    unlabIdx = find(labels == 0);
    labels(unlabIdx) = [];
    feats(unlabIdx,:) = [];
    
    % Pseudo-train N.N.
    model = fitcknn(feats, labels, 'NumNeighbors', 5);
	[predictions, scores] = predict(model, featsTest);
	scores = sort(scores, 'descend');
	
end

function [predictions, scores] = predictUsingGPU2(model, unlab, f)
	
% 	predictions = length(unlab.labels);
% 	scores = zeros( length(unlab.labels), 2 );
	
	% Predict for each unlabeled data point
% 	for unlabInst = 1 : length(unlab.labels)
		
		dd = model.D( unlab.Idx(unlabInst), : );
		
		% Find NN labels
		NNs = zeros( model.K, 1 );
		
		% Get rid of unlab instances
		for i = 1 : length(model.labels)
			if(model.labels(i) == 0)
				dd(i) = +Inf;
			end
		end
		
		for k = 1 : model.K
			[~, I] = min( dd );
			NNs(k) = model.labels( I );
			dd(I) = +Inf;
		end
		
		% Decide on label and scores
		numPos = nnz( NNs == 1 );
		numNeg = nnz( NNs == -1);
		
		if(numPos > numNeg)
			predictions(unlabInst) = 1;
			scores(unlabInst, 1) = (1/5) * numPos;
			scores(unlabInst, 2) = (1/5) * numNeg;
		else
			predictions(unlabInst) = -1;
			scores(unlabInst, 2) = (1/5) * numPos;
			scores(unlabInst, 1) = (1/5) * numNeg;
		end
% 	end
	
	
end

function [predictions, scores] = predictUsingGPU1(model, unlabFeats)

	labeledFeats = model.feats;
	labeledLabels = model.labels;
	
	predictions = zeros( size(unlabFeats, 1), 1);
	scores = zeros( size(unlabFeats, 1), 2);

	% Select GPU
	gpuDevice(1);
	
	% Predict for each unlabeled data point
	for unlabInst = 1 : size(unlabFeats, 1)
		unlabFeat = unlabFeats(unlabInst, :);
		
		% Get distance
		Glab = gpuArray(labeledFeats);
		Gunlab = gpuArray(unlabFeat);
		
		df = bsxfun(@minus, Glab, Gunlab);
		df = abs(df);
		dd = df * df';
		dd = sqrt( complex(dd) );
		dd = diag(dd);
		
		dd = gather(dd);
		
		% Find NN labels
		NNs = zeros( model.K, 1 );
		
		for k = 1 : model.K
			[~, I] = min( dd );
			NNs(k) = labeledLabels( I );
			dd(I) = +Inf;
		end
		
		% Decide on label and scores
		numPos = nnz( NNs == 1 );
		numNeg = nnz( NNs == -1);
		
		if(numPos > numNeg)
			predictions(unlabInst) = 1;
			scores(unlabInst, 1) = (1/5) * numPos;
			scores(unlabInst, 2) = (1/5) * numNeg;
		else
			predictions(unlabInst) = -1;
			scores(unlabInst, 2) = (1/5) * numPos;
			scores(unlabInst, 1) = (1/5) * numNeg;
		end
			
		
	end


end



















