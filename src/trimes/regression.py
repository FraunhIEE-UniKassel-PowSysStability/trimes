import pandas as pd
import numpy as np
from typing import Union


def polyfit(ts:Union[pd.DataFrame, pd.Series], degree:int, t_offset=None) -> np.array:
  """Polynomial fit using numpy's 'polyfit'.
  
  Args:
      ts (Union[pd.DataFrame, pd.Series]): Values for polynomial fit (x=ts.index, y=ts.to_numpy())
      degree (int): Polynomial 
      t_offset (bool, optional): time offset (x=ts.index+t_offset)

  Returns:
      np.array: Polynomial coeficients

  It was tried to implement this function using the more recent polynomial approximation as recommended by numpy as shown below. However, there were problems when 'ts' is a DataFrame.
   
  if not full:
    return np.polynomial.polynomial.Polynomial.fit(ts.index, ts.to_numpy(), degree).convert().coef
  else: 
    return np.polynomial.polynomial.Polynomial.fit(ts.index, ts.to_numpy(), degree)
  """
  if not t_offset:
    return np.polyfit(ts.index, ts.to_numpy(), degree)
  else:  
    return np.polyfit(ts.index+t_offset, ts.to_numpy(), degree)
 

