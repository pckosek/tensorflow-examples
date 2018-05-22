import numpy as np
import matlab.engine

# --------------------------------------------------- #
# MATLAB CONNECTION CLASS
# --------------------------------------------------- #
class matlab_connection:

  def __init__(self, session_name):
    self.session_name = session_name
    self.eng = matlab.engine.connect_matlab(session_name)

  def put_var(self, var_name, values) :
    self.eng.workspace[var_name] = matlab.double( np.asarray(values).tolist() ) 

  def get_var(self, var_name) :
    return np.asarray( eng.workspace[var_name] )