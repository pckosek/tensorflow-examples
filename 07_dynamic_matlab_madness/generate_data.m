n = 10000; 			% length of sequence
foo = zeros(1,n); 	% initialize sequence

npts = 75;			% num pts to be stuffed with random numbers
foo( randi([1, n], [1, npts] ) ) = rand( 1, npts);

wLen = 1250;
win  = hann(wLen)';
foo_conv = conv( foo, win, 'same');

foo_conv = foo_conv./max(foo_conv);
amp = 0.3;
off = 0.7;

foo_conv = amp*foo_conv + off;

x = cos( 2*pi*2*[0:n-1]/1000 ) .* foo_conv;
y = cos( 2*pi*2*[0:n-1]/1000 - pi/2 ) .* foo_conv;

seq_length = 15;
data_dim = 5;


trainX = zeros(800, seq_length, data_dim);
testX  = zeros(800, seq_length, data_dim);

trainY = zeros(800, 1);
testY  = zeros(800, 1);

trainI = zeros(800, 1); trainI(1) = 1;
testI  = zeros(800, 1); testI(1)  = 1;


selection_matrix = [0:data_dim-1]+[0:seq_length-1]';

lookahead = 5;

for indx = 1:800
	trainX(indx,:,:) = x(indx+selection_matrix);
	trainY(indx) = y( max(max(indx+selection_matrix)) + lookahead);
end

testingOffset = 1000;
for indx = 1:800
	testX(indx,:,:) = x(indx+selection_matrix + testingOffset);
	testY(indx) = y( max(max(indx+selection_matrix)) + lookahead + testingOffset);
end