[b,a] = butter(7, .2);

%------------------------% 
n=5000;

trainX = zeros(n,100,1);
trainY = zeros(n,100,1);

for indx = 1:n
    trainX(indx,:,:) = [ zeros(1,20), 2*rand(1,80)-1 ];
	trainY(indx,:,:) = filter(b,a, trainX(indx,:,:) );
end


%------------------------% 
n=10;
testX = zeros(n,100,1);
testY = zeros(n,100,1);

for indx = 1:n
    testX(indx,:,:) = [ zeros(1,20), 2*rand(1,80)-1 ];
	testY(indx,:,:) = filter(b,a, testX(indx,:,:) );
end

