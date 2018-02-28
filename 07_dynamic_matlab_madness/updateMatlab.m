function a_out = updateMatlab()

evalStr = 'figure(f1); plot( reshape(H(1,1,1,:), [10,1] ), ''k+-'');figure(f2); plot( reshape(H(1,2,1,:), [10,1] ), ''k+-'');figure(f3); plot( reshape(H(2,1,1,:), [10,1] ), ''k+-'');figure(f4); plot( reshape(H(2,2,1,:), [10,1] ), ''k+-'')';
evalin('base', evalStr);

a_out = true;