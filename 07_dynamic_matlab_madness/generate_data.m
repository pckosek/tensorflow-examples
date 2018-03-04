n = 1000;    			% length of sequence

sequenceLen = 13; 		% make it a prime number

defaultMatrix = [0:sequenceLen-1]; 

trainX = zeros(n,sequenceLen);
trainY = zeros(n,1);

for indx = 1:n
	trainX(indx,:) = indx + defaultMatrix ;
	trainY(indx)   = indx + sequenceLen;
end