% democraticCo_predict.m
%
% Kemal Tugrul Yesilbek
% Sept 2015
%
% Predicts test instances' labels based on ensemble of models learned by
% democratic Co-learning. Practices majority voting
%
% =================
% How to use:
% 1. Pass bundle returned from democraticCo_train.m and the test features
% 2. Change prediction mechanism if you changed the models in the training
% script
%
% =================
% IN:
% bundle: Bundle returned from training script
% testFeats: Test features, each row represents an instance
%
% =================
% predictions: Labels predicted for test features
%

function [ predictions ] = democraticCo_predict( bundle, testFeats )

	predictions = zeros( size(testFeats, 1), 1);

	for inst = 1 : size(testFeats, 1)
		[pred(1), ~] = predictDTree_Democ( bundle{1}.model, testFeats(inst, :) );
		[pred(2), ~] = predictLinearSVM_Democ( bundle{2}.model, testFeats(inst, :) );
		%[pred(2), ~] = predictNB_Democ( bundle{2}.model, testFeats(inst, :) );
		[pred(3), ~] = predictNN_Democ_test( bundle{3}.model, testFeats(inst, :) );

		predictions(inst) = mode(pred); % Majority voting
	end


end







































