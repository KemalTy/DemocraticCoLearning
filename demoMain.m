% demoMain.m
%
% Tests and demonstrates the implementation of Democratic Co-Learning
%
% Kemal Tugrul Yesilbek
% August 2015
%

%% Initialize
close all;
clear all;
clc;
workspace;
fprintf('Script Start...\n');

%% Options
featsDir = 'feats.mat';
labelsDir = 'labels.mat';

%% Load Data
load(featsDir);
load(labelsDir);

%% Dataset dependent fixs and arrangements
% Make data binary
labels( find(labels > 2) ) = 0;
labels( find(labels == 1) ) = 1;
labels( find(labels == 2) ) = -1;

% Make positive instances minority
numPosInst = 10;
numNegInst = 300;

if(length(find(labels == -1)) > numNegInst)
   winnowNo = length(find(labels == -1)) - numNegInst;
   negIdx = find(labels == -1);
   winnowLocal = randperm(length(negIdx), winnowNo);
   winnowGlobal = negIdx(winnowLocal);
   labels(winnowGlobal) = 0;
end

if(length(find(labels == 1)) > numPosInst)
   winnowNo = length(find(labels == 1)) - numPosInst;
   posIdx = find(labels == 1);
   winnowLocal = randperm(length(posIdx), winnowNo);
   winnowGlobal = posIdx(winnowLocal);
   labels(winnowGlobal) = 0;
end

% Shuffle instance indexes
labelsOld = labels;
featsOld = feats;
newIdx = randperm(length(labels), length(labels));
feats = []; labels = [];

for i = 1 : length(newIdx)
	labels(i) = labelsOld(newIdx(i));
	feats(i,:) = featsOld(newIdx(i), :);
end


%% Visualize initial distribution
visualize2Ddist( feats, labels );

%% Run Democratic Co-Learning Training
trainOptions.numOfLearners = 3;
trainOptions.maxIter = 100;
trainOptions.isDebug = false;

[bundle] = democraticCo_train(feats, labels, trainOptions);

%% Test the prediction system

% Generate some new syntetic data
testFeats = [];
for i = 1 : 3 : 120
	for j = 1 : 3 : 120
		testFeats = [testFeats ; i, j];
	end
end

% Predict on test data
pred = democraticCo_predict(bundle, testFeats);

%% Visualize Test Distributions
visualize2Ddist( testFeats, pred );
title('Predictions on Test Data');

% Bye
fprintf('Script End...\n');















