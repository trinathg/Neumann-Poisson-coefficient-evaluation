import scipy.sparse.linalg as splinal
import math
from scipy.sparse import hstack
import scipy.sparse as sp
import numpy as np
import time


for i in range(10):
	ReR = sp.lil_matrix((10,10))    
	
