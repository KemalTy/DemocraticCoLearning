function visualize2Ddist( feats, labels )

% Decide on colors
% uniqueLabels = unique(labels);
% C = length( uniqueLabels );
% colors = lines(C);

% Create new figure
figure; hold on; grid on; grid minor;

for i = 1 : length(labels)
    
    %colorIdx = find( uniqueLabels == labels(i) );
    %color = colors( colorIdx, : );
    if(labels(i) == 0)
        plot( feats(i,1), feats(i,2), 'k.');
    elseif(labels(i) == 1)
        plot( feats(i,1), feats(i,2), 'go');
    elseif(labels(i) == -1)
        plot( feats(i,1), feats(i,2), 'rx');
    else
        plot( feats(i,1), feats(i,2), 'cd');
    end
    
end

hold off;

end

