function visualizeDecisionBoundaries( bundle )

    % Generate grid points
    pnts = [];
    for i = 1 : 3 : 120
        for j = 1 : 3 : 120
            pnts = [pnts ; i, j];
        end
	end
	
	% Get predictions from diff. classifiers and visualize them
    [pred, ~] = predictDTree_Democ(bundle{1}.model, pnts);
    visualize2Ddist(pnts, pred);
    title('Decision Boundary: DTree');
	
	% Get predictions from diff. classifiers and visualize them
    [pred, ~] = predictLinearSVM_Democ(bundle{2}.model, pnts);
	%[pred, ~] = predictNB_Democ(bundle{2}.model, pnts);
    visualize2Ddist(pnts, pred);
    title('Decision Boundary: NB');
    
    % Get predictions from diff. classifiers and visualize them
    [pred, ~] = predictNN_Democ_test(bundle{3}.model, pnts);
    visualize2Ddist(pnts, pred);
    title('Decision Boundary: NN');
    
    
    
    
    
end

