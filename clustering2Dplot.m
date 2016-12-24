function [] = clustering2Dplot(X,clusters,means)
[n,d] = size(X);
k = max(clusters);

clf;hold on;
colors = getColorsRGB;
symbols = {'s','o','v','^','x', '+', '*', 'd', '<', '>', 'p'};
if isempty(clusters)
    plot(X(:,1),X(:,2),'o','Color',[.75 .75 .75],'MarkerFaceColor','k');
else
    for j = 1:k
        plot(X(clusters==j,1),X(clusters==j,2),'o','Color',.75*colors(j,:),...
            'Marker',symbols{j},'MarkerSize',5,'MarkerFaceColor',.75*colors(j,:));
    end
end
for j = 1:k
    plot(means(j,1),means(j,2),'o','Color',colors(j,:),...
        'Marker',symbols{j},'MarkerSize',12,'MarkerFaceColor',colors(j,:));
end
pause(.1);
end
