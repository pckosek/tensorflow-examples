
fname = 'animation.gif';

f1 = figure;
for n = 1:40
    plot([reshape(y_tests(n,:,:),s),reshape(y_preds(n,:,:),s)],'+-');
    set(gca,'ylim',[0,1]);
    drawnow
    
    title( sprintf('Test Sequence #%.f', n ) );
    ylabel('output')
    xlabel('step')
    
    frame = getframe(f1);
    im = frame2im(frame);
    [X,map] = rgb2ind(im,256);
    
    frame = getframe(f1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    if n == 1
        imwrite(X,map,fname,'gif', 'Loopcount',inf, 'DelayTime', .30);
    else
        imwrite(X,map,fname,'gif','WriteMode','append', 'DelayTime', .30);
    end
end
