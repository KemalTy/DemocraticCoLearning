function [prediction, scores] = predictNN_Democ_train( model, unlab, unlabIdx )

	% Get corresponding distance vector
	dd = model.D( unlab.Idx(unlabIdx), : );

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
		prediction = 1;
		scores(1) = (1/5) * numPos;
		scores(2) = (1/5) * numNeg;
	else
		prediction = -1;
		scores(2) = (1/5) * numPos;
		scores(1) = (1/5) * numNeg;
	end
	
end















