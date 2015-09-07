function visualizeDistWithConf( feats, labels, conf, pred )

% Color decisions
colors = zeros(length(labels), 3);

for i = 1 : length(labels)
    
    if(labels(i) == 0)
        colors(i,:) = [0.12, 0.11, 0.13];
    elseif(labels(i) == 1)
        colors(i,:) = [0.05, 0.95, 0.05];
    elseif(labels(i) == -1)
        colors(i,:) = [0.85, 0.10, 0.15];
    else
        colors(i,:) = [0, 0.4470, 0.7410];
    end
    
end

% Create new figure
fig = figure('DeleteFcn','doc datacursormode');
hold on; grid on; grid minor;
scatter( feats(:,1), feats(:,2), [], colors);

% Set new labels
dcm_obj = datacursormode(fig);
set(dcm_obj,'UpdateFcn',{@myupdatefcn, conf, pred});

hold off;



end


function txt = myupdatefcn(~,event_obj, conf, pred)
	% Customizes text of data tips
	I = get(event_obj, 'DataIndex');
	
	I = mod(I, size(conf,1));
	
	txt = {['I: ',num2str(I)],...
		   ['DTree: ',num2str(conf(I,1,1)), ' Pred: ' num2str(pred(I, 1))],...
		   ['SVM: ',num2str(conf(I,2,1)), ' Pred: ' num2str(pred(I, 2))],...
		   ['KNN: ',num2str(conf(I,3,1)), ' Pred: ' num2str(pred(I, 3))]
		   };
end




























