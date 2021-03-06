[b,a] = butter(7, .2);

%------------------------% 
x1 = [zeros(1,40), ones(1,60)];
y1 = filter(b,a,x1);

x2 = [zeros(1,40), -ones(1,60)];
y2 = filter(b,a,x2);

trainX = zeros(2,100,1);
trainY = zeros(2,100,1);

trainX(1,:,:) = x1;
trainX(2,:,:) = x2;

trainY(1,:,:) = y1;
trainY(2,:,:) = y2;

%------------------------% 
testX = zeros(6,100,1);
testY = zeros(6,100,1);

testX(1,:,:) = [zeros(1,40), ones(1,60).*rand(1,60)];
testX(2,:,:) = [zeros(1,40), -ones(1,60).*rand(1,60)];
testX(3,:,:) = [zeros(1,20), ones(1,80)];
testX(4,:,:) = [zeros(1,20), -ones(1,80)];
testX(5,:,:) = [zeros(1,20), .5*ones(1,80)];
testX(6,:,:) = zeros(1,100); 
    testX(6,30:35,1) = 1;
    testX(6,60:65,1) = 1;

for indx = 1:6
	testY(indx,:,:) = filter(b,a, testX(indx,:,:) );
end