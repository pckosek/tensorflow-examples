backprops = [15:30];
indx = find(backprops==backprop);

out(indx).losses = losses;
out(indx).y_preds = y_preds(:,end);
