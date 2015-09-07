% democraticCo_train.m
%
% Kemal Tugrul Yesilbek
% August 2015
%
% Implementation of democratic Co-Learning's training phase
%
% ================================
% How to Use:
%
% 1. Decide on options and pass as an input along features and labels
%
% 2. If it will be a debug run, use 2D features and following labeling
% setting:
% 0 : Unlabeled
% 1 : Positive class
% -1: Negative class
%
% 3. Original learners are: 5-KNN, RBF SVM, and Decision Tree. Change them
%	  if neccessary, beware of their order while changing.
%
% 4. Add functions to the Matlab path
%
% 5. I introduced some changings in the algorithm in accepting to Li' and
% removed  qi' > qi criteria. Change them if needed.
%
% 6. Implementation is based on binary classification
%
% ================================
% IN:
%
% feats: Features, including unlabeled points. Each row should contrain a
% single instance
%
% labels: Label pairs of features 
%
% options: Options for training
% -- numOfLearners : How many learners will there be
% -- maxIter : Maximum iteration of the main loop
% -- isDebug : Is this a debug run? If it is so there will be lots of
%					output!
%
% ================================
% OUT:
% out: Bundle of learners' models, final labels, features, error rates
%


function [ out ] = democraticCo_train( feats, labels, options )
fprintf('Democratic Co-Learning (Train) start...\n');

%% Options
numOfLearners = options.numOfLearners;
maxIter = options.maxIter;
isDebug = options.isDebug;


%% Initialize variables
unlab.Idx = find(labels == 0);
unlab.labels = labels(unlab.Idx);
unlab.feats = feats(unlab.Idx, :);


for i = 1 : numOfLearners
   
   % Initialize labeled data sets for all learners
   bundle{i}.labels = labels;
   bundle{i}.feats = feats;
   bundle{i}.possibleNewLabels = [];
   
   % Initialize the mislabeled estimation for all learners
   bundle{i}.e = 0;
   
end


%% Largest loop
iterNo = 0;
bundlePrev = bundle;

while( ~stopCriteria( bundlePrev, bundle, iterNo, maxIter ) )
    
    % Increment the loop
    iterNo = iterNo + 1;
    bundlePrev = bundle;
    fprintf('*** Iter: %d...\n', iterNo);
    
    % Run learners on their training sets
    bundle = learnModels( bundle );
    
	% Visualize the decision boundaries (x,y=0:120)
	if(isDebug)
		visualizeDecisionBoundaries( bundle );
	end
	
    % Single iteration of demo.co.learning
    [bundle, conf, pred] = singleRun(bundle, unlab, isDebug);
	
    % Debug outputs
    if(isDebug)
       for learnerIdx = 1 : numOfLearners
		   
		   % Re-arrange pred and arrange for dist with conf as pred and
		   % conf are only for unlabeled ones
		   for inst = 1 : length(bundle{learnerIdx}.labels)
			   if(~ismember(inst, unlab.Idx) ) % Is not a member of Unlab
				   pred_(inst,:) = [0 0 0];
				   conf_(inst,1,1) = 0;
				   conf_(inst,2,1) = 0;
				   conf_(inst,3,1) = 0;
			   else
				   pred_(inst, :) = pred( find(unlab.Idx == inst), : );
				   conf_(inst,1,1) = conf( find(unlab.Idx == inst), 1, 1 );
				   conf_(inst,2,1) = conf( find(unlab.Idx == inst), 2, 1 );
				   conf_(inst,3,1) = conf( find(unlab.Idx == inst), 3, 1 );
			   end
		   end
		   
		   % Visualize new distribution
           visualizeDistWithConf(bundle{i}.feats, bundle{i}.labels, conf_, pred_);
           title(['Dist. at Iter: ' num2str(iterNo) 'For: ' num2str(learnerIdx)]);
       end
       
    end
    
end % EO while(changes) loop

% Return models, and their stuff in a cell
out = bundle;

end

% Let trained classifiers predict labels on unlabeled data and add the ones
% that satisfy the constraints
function [bundle, conf, pred] = singleRun(bundle, unlab, isDebug)
    
    fprintf('Single Run...\n');

    %% Reset bundle
    for i = 1 : length(bundle)
       bundle{i}.possibleNewLabels = [];
    end
    
    % For each unlabeled instance, predict on and propose instances to add new labels
    for unlabIdx = 1 : length(unlab.labels)
        %fprintf('Unlabeled %d of %d...\n', unlabIdx, length(unlab.labels));
        
        %% Predict with all learners
        [pred(unlabIdx, 1), conf(unlabIdx, 1, :)] = predictDTree_Democ( bundle{1}.model, unlab.feats(unlabIdx, :) );
		[pred(unlabIdx, 2), conf(unlabIdx, 2, :)] = predictLinearSVM_Democ( bundle{2}.model, unlab.feats(unlabIdx, :) );
		%[pred(unlabIdx, 2), conf(unlabIdx, 2, :)] = predictNB_Democ( bundle{2}.model, unlab.feats(unlabIdx, :) );
        %[pred(unlabIdx, 3), conf(unlabIdx, 3, :)] = predictNN_Democ_train( bundle{3}.model, unlab.feats(unlabIdx, :),  unlab, true);
		[pred(unlabIdx, 3), conf(unlabIdx, 3, :)] = predictNN_Democ_train( bundle{3}.model, unlab, unlabIdx);
        
        %% Decide on majority vote
        majClass(unlabIdx) = mode( pred(unlabIdx,:) );
        
        % Is there a majority vote after all? (First criteria)
		numMaj = nnz( pred(unlabIdx,:) == majClass(unlabIdx) );
		numMin = nnz( pred(unlabIdx,:) ~= majClass(unlabIdx) );
        if( numMaj <= numMin )
            continue;
        end
        
        %% Calculate confidence intervals
        % First index in confidences is always the predicted class
        [majConfInterval(unlabIdx), minConfInterval(unlabIdx)] = calculateConfidence(pred(unlabIdx, :), conf(unlabIdx, :, :));
        
        %% Decide if this prediction should be added to the labeled ones
        % I, here, will do a modification to the algorithm. It does not
        % make sense to me when all of the classifiers vote for the
        % majority class and non of them get a new label. I will change
        % this mechanism and will add the new label to all Li' if they all
        % aggreed on the same class.
        if( nnz( pred(unlabIdx,:) == majClass(unlabIdx)) == length(bundle) )
            for clIdx = 1 : length(bundle)
                bundle{clIdx}.possibleNewLabels = [bundle{clIdx}.possibleNewLabels, unlabIdx];
                %fprintf('A possible new label for learner:%d ...\n', clIdx);
            end
        end
        
        if(majConfInterval(unlabIdx).mean > minConfInterval(unlabIdx).mean)
            % Add possible new labels to classifiers that voted for
            % minority
            for clIdx = 1 : length(bundle)
                if(pred(unlabIdx,clIdx) ~= majClass)
                   bundle{clIdx}.possibleNewLabels = [bundle{clIdx}.possibleNewLabels, unlabIdx];
                   %fprintf('A possible new label for learner:%d ...\n', clIdx);
                end
            end
        end
        
        % Debug prints
        if(isDebug)
            fprintf('Majority Class: %d...\n', majClass(unlabIdx));
            fprintf('NN Conf: %d DTree Conf: %d SVM Conf: %d...\n', conf(unlabIdx, 3, 1) , conf(unlabIdx, 1, 1), conf(unlabIdx, 2, 1));
            fprintf('Maj. Conf. Mean: %d Lower: %d Upper:%d\n', majConfInterval(unlabIdx).mean, majConfInterval(unlabIdx).lower, majConfInterval(unlabIdx).higher);
            fprintf('min. Conf. Mean: %d Lower: %d Upper:%d\n', minConfInterval(unlabIdx).mean, minConfInterval(unlabIdx).lower, minConfInterval(unlabIdx).higher);
        end
        
        
	end
	
	% Plot confidences of the classifiers
	if(isDebug)
		figure; hold on;
		plot(conf(:,1,1), 'rx');
		plot(conf(:,2,1), 'go');
		plot(conf(:,3,1), 'bd');
	end
    
    %% Estimate if possible new labels will improve the accuracy
    for classifierIdx = 1 : length(bundle)
        sizeLi = length(find(bundle{classifierIdx}.labels ~= 0));
        sizeLiP = length(bundle{classifierIdx}.possibleNewLabels);
        
        qi = sizeLi * (1 - 2 * (bundle{classifierIdx}.e / sizeLi))^2;
        eip = (1 - (calculateClassifierConfidence(conf, classifierIdx))) * sizeLiP;
        qip =  (sizeLi + sizeLiP) * ( 1 - (2*((bundle{classifierIdx}.e + eip)/(sizeLi+sizeLiP)) ) )^2;
        
        if(isDebug)
            fprintf('|Li|:%d |Li_p|:%d...\n', sizeLi, sizeLiP);
            fprintf('qi: %d ei_p: %d qi_p: %d...\n', qi, eip, qip);
        end
        
        %% Add to labeled ones if qip > qi
        if(true)%(qip > qi) % Just accept them all!!!!
           for i = 1 : length(bundle{classifierIdx}.possibleNewLabels)
              bundle{classifierIdx}.labels(  unlab.Idx(bundle{classifierIdx}.possibleNewLabels(i)) ) = majClass( bundle{classifierIdx}.possibleNewLabels(i) );
           end
           
           % Reassign error estimation
           bundle{classifierIdx}.e = bundle{classifierIdx}.e + eip;
        end
        
    end
    
    
    
end

% Calculate classifiers' average lower, upper, and mean confidence levels
function [lower, upper, mean_] = calculateClassifierConfidence(conf, classifierIdx)
    meanC = mean( conf(:, classifierIdx, 1) );
    stdC = std( conf(:, classifierIdx, 1) );
    alpha = -1.96;
    N = size(classifierIdx, 1);
    margOfErr = alpha * (stdC / sqrt(N));
    intervals(1) = meanC + margOfErr;
    intervals(2) = meanC - margOfErr;
    
    lower = min(intervals);
    upper = max(intervals);
    mean_ = mean(intervals);
end

% Calculate classifiers'  lower, upper, and mean confidence levels on
% majority and minority instances
function [majConfInterval, minConfInterval] = calculateConfidence(pred, conf)
    
    % Decide on majority vote
    majClass = mode( pred );
    
    % Get confidences of majority and minority decided classifiers
    majConf = []; minConf = [];
    
    for predIdx = 1 : length(pred)
       if(pred(predIdx) == majClass)
           majConf = [majConf, conf(1, predIdx, 1)];
       else
           minConf = [minConf, conf(1, predIdx, 1)];
       end
    end
    
    % Calculate confidence interval of majority class
    maj_mean = mean(majConf);
    maj_std = std(majConf);
    alpha = -1.96;
    N = length(majConf);
    maj_margOfErr = alpha * (maj_std / sqrt(N));
    
    intervals(1) = maj_mean + maj_margOfErr;
    intervals(2) = maj_mean - maj_margOfErr;
    
    majConfInterval.lower = min(intervals);
    majConfInterval.higher = max(intervals);
    majConfInterval.mean = mean(intervals);
    
    % Calculate confidence interval of minority class
    if(isempty(minConf))
        minConfInterval.lower = 0;
        minConfInterval.higher = 0;
        minConfInterval.mean = 0;
    else
        min_mean = mean(minConf);
        min_std = std(minConf);
        alpha = -1.96;
        N = length(minConf);
        min_margOfErr = alpha * (min_std / sqrt(N));

        intervals(1) = min_mean + min_margOfErr;
        intervals(2) = min_mean - min_margOfErr;

        minConfInterval.lower = min(intervals);
        minConfInterval.higher = max(intervals);
        minConfInterval.mean = mean(intervals);
    end
    
end


% Train learners. Watch the arrangements of the learners!!
function [bundle] = learnModels( bundle )

    bundle{1}.model = trainDTree_Democ(bundle{1}.labels, bundle{1}.feats);
    bundle{2}.model = trainLinearSVM_Democ(bundle{2}.labels', bundle{2}.feats);
	%bundle{2}.model = trainNB_Democ(bundle{2}.labels, bundle{2}.feats);
    bundle{3}.model = trainNN_Democ(bundle{3}.labels, bundle{3}.feats);

end


% Checks if the Ls from previous iteration is same in current iteration
function [isStop] = stopCriteria( bundle_prev, bundle_curr, iterNo, maxIter )
    
    % Let it run for first iter
    if(iterNo == 0)
        isStop = false;
        return;
    end
    
    % Iteration limit
    if(iterNo > maxIter)
        isStop = true;
        fprintf('Reached to Maximum Iteration...\n');
        return;
    end

    % Algorithmic Criteria
    isStop = true;
    for i = 1 : length(bundle_curr)
       if( ~issame( bundle_prev{i}.labels, bundle_curr{i}.labels ) )
           isStop = false;
           return;
       end
    end
    fprintf('Algorithmic Stopping Criteria Reached...\n');
    
end





























