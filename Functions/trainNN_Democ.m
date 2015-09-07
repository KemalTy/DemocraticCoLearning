% Pseudo-Trains N.N on data
function [ model ] = trainNN_Democ( labels, feats )
	fprintf('Training KNN...\n');
    model = trainUsingGPU(labels, feats);

end


function [model] = trainUsingMatlab(labels, feats)
% Remove unlabeled instances
    unlabIdx = find(labels == 0);
    labels(unlabIdx) = [];
    feats(unlabIdx,:) = [];
    
    % Pseudo-train N.N.
    model = fitcknn(feats, labels, 'NumNeighbors', 5);
end

function [model] = trainUsingGPU(labels, feats)

	% During democ. co learning training, no new instance is introduced.
	% So, a single pairwise distance calculation for all instances will do
	% the work as those distances will not change but the instances'
	% labels.
	model.D = getAllDist( feats );
	
	% Register to model bundle
	model.labels = labels;
	model.feats = feats;
	model.K = 5;
	
	

end

function [D] = getAllDist(feats)
	
D = squareform( pdist(feats) );

% 	% Select GPU
% 	gpuDevice(1);
% 	
% 	% Get distance
% 	G = gpuArray(feats);
% 	
% 	df = bsxfun(@minus, Glab, Gunlab);
% 	df = abs(df);
% 	dd = df * df';
% 	dd = sqrt( complex(dd) );
% 	dd = diag(dd);
% 	
% 	dd = gather(dd);
	
	
	
end


























