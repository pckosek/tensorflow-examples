f1 = figure; 
plot( losses );

title( 'Clean Losses' )
ylabel('loss')
xlabel('training step')


frame = getframe(f1);
im = frame2im(frame);
[X,map] = rgb2ind(im,256);

imwrite(X,map,'loss.jpg','jpg');