n  = 5000;    			% length of sequence
Fs = 1000;

sequenceLen = 13; 		% make it a prime number

defaultMatrix = [0:sequenceLen-1]; 


x = .5 + .5*cos( 2*4*pi*[0:n-1]/Fs );
y = .5 + .5*sin( 2*8*pi*[0:n-1]/Fs );

trainX = zeros(n,sequenceLen);
trainY = zeros(n,1);

for indx = 1:n-sequenceLen
	trainX(indx,:) = x( indx + defaultMatrix );
	trainY(indx)   = y( indx + sequenceLen );
end