function a_out = should_stop(varargin)
% tell python (tensorflow) whether or not it should stop
persistent retVal;

if ~exist('retVal','var')
    retVal = false;
end

if nargin > 1
    error('too many inputs');
elseif (nargin == 1)
    
    inVal = varargin{1};
    
    if length( inVal ) > 1
        error( 'too many inputs' );
    else
        if ~islogical(inVal) 
            error('input must be a logical value');
        else
            retVal = inVal;
        end
    end
end

a_out =  retVal;