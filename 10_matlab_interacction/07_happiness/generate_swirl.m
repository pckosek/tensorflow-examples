raw = normalize( cumsum( randn( 100, 128 ) ) );

y = filter(b,a,raw);
w = normalize(hamming(100)')*ones(1,128);

z = w.*y;


nx = 301;
len = (nx+1)*50;

ch_1 = zeros(1,len);
ch_2 = zeros(1,len);

for n = 1:nx
    this_i = randi( [1, 128], 1 );
    this_j = randi( [1, 128], 1 );
    
    ch_1 = overlap( ch_1, z(:,this_i)', 1+(n-1)*50 );
    ch_2 = overlap( ch_2, z(:,this_j)', 1+(n-1)*50 );
end

validLength = (nx-1)*50;

ch_1 = ch_1( 51 + [0:validLength-1] );
ch_2 = ch_2( 51 + [0:validLength-1] );

ch = [ch_1;ch_2]';

nCases = validLength - 100;

trainX = zeros(nCases-1,100,2);
trainY = zeros(nCases-1,100,2);

s = [1:100];
for indx = 0:nCases-1-1
    trainX(indx+1,:,:) = ch( indx+1+s, :);
    trainY(indx+1,:,:) = ch( indx+1+1+s, :);
end

testX = trainX;
testY = trainY;