numSets = 9950;

inputTimeSteps  = 30;
outputTimeSteps = 30;

inputVectorSize  = 1;
outputVectorSize = 1;
% split data into training and testing

master_sequence = [0:100,99:-1:1]./100;
target_sequence = 1 - [0:100,99:-1:1]./100;
n               = length(master_sequence);

selectVect = [0:inputTimeSteps-1];

X = zeros(numSets,inputTimeSteps,inputVectorSize);
Y = zeros(numSets,outputTimeSteps,outputVectorSize);

for indx = 1:numSets
    
    inSeq  = master_sequence( mod( selectVect+indx+0,n ) +1 );
    outSeq = target_sequence( mod( selectVect+indx+40,n ) +1 );
    
    scaleFactor = 1;
    this_x = scaleFactor * inSeq;
    this_y = scaleFactor * outSeq;
    
    X(indx,:,:) = reshape( this_x, [1, inputTimeSteps, 1]);
    Y(indx,:,:) = reshape( this_y, [1, outputTimeSteps, 1]);
end

splitR = 0.8;
splitBelow = floor( numSets*splitR );

% TRAINING DATA
trainX = X( 1:splitBelow, :, : );
trainY = Y( 1:splitBelow, :, : );

% TESTING DATA
testX = X( splitBelow+1:end, :, : );
testY = Y( splitBelow+1:end, :, : );

% n = 10;
% trainX = zeros(n,2,1);
% trainY = trainX;
% 
% for indx = 1:n
%     
%     trainX(indx,1,1) = indx/10;
%     trainX(indx,2,1) = 0;
%     trainY(indx,1,1) = 0;
%     trainY(indx,2,1) = indx/10;
% end
% testX = trainX;
% testY = trainY;
% 
