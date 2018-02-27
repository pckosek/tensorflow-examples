function a_out = generate_data()
% load the data
load fisheriris

% count the measurements
n = length(species);

% create labels
labels = zeros(n,1);

for indx = 1:n
    labels(indx) = findIrisLabel( species{indx} );
end

% split data into training and testing
splitR = 0.8;
splitBelow = floor( n*splitR );

trainX = meas( 1:splitBelow, : );
trainY = int16( labels( 1:splitBelow ) );

testX = meas( splitBelow+1:end, : );
testY = int16( labels(splitBelow+1:end ) );

% this is a little funky; a hack to 'return' vars
% for later on
assignin('base', 'trainX', trainX);
assignin('base', 'trainY', trainY);
assignin('base', 'testX', testX);
assignin('base', 'testY', testY);

a_out = true;
end
%  END OF OPERATIONS


function labelIndex = findIrisLabel( str_in )
labSet = {'setosa', 'versicolor', 'virginica' };

labelIndex = -1;
for indx = 1:length(labSet)
    if strcmp( str_in, labSet(indx) )
        labelIndex = indx-1;
        break;
    end
end

end
%-------------EOF
    