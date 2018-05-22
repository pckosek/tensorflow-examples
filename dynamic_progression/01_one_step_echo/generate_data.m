numSets = 10000;

inputTimeSteps  = 5;
outputTimeSteps = 5;

inputVectorSize  = 1;
outputVectorSize = 1;
% split data into training and testing

X = zeros(numSets, inputTimeSteps, inputVectorSize );
Y = zeros(numSets, outputTimeSteps, outputVectorSize );

for indx = 1:numSets
    this_x = randi( [1,100], [5,1,1] )./100;
    this_y = [this_x; sum(this_x)*ones(3,1,1)];
    
    X(indx,:,:) = this_x;
    Y(indx,:,:) = this_x;
end

splitR = 0.8;
splitBelow = floor( numSets*splitR );

% TRAINING DATA
trainX = X( 1:splitBelow, :, : );
trainY = Y( 1:splitBelow, :, : );

% TESTING DATA
testX = X( splitBelow+1:end, :, : );
testY = Y( splitBelow+1:end, :, : );

n = 10;
trainX = zeros(n,2,1);
trainY = trainX;

for indx = 1:n
    
    trainX(indx,1,1) = indx/10;
    trainX(indx,2,1) = 0;
    trainY(indx,1,1) = 0;
    trainY(indx,2,1) = indx/10;
end
testX = trainX;
testY = trainY;

