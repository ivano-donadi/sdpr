import numpy as np
def get_lost_indices():
  losts = []
  losts.extend([x for x in range(587,1210)])
  losts.extend([x for x in range(2320,3054)])
  losts.extend([x for x in range(4203,4360)])
  losts.extend([x for x in range(4850,5250)])
  losts.extend([x for x in range(5460,6050)])
  losts.extend([x for x in range(6360,6490)])
  losts.extend([x for x in range(6750,7000)])
  losts.extend([x for x in range(7150,7450)])
  losts.extend([x for x in range(7895,8200)])
  losts.extend([x for x in range(8647,8786)])
  losts.extend([x for x in range(9045,9158)])
  losts.extend([x for x in range(9521,9575)])
  losts.extend([x for x in range(10190,10860)])
  losts.extend([x for x in range(11111,11470)])
  losts.extend([x for x in range(11840,11870)])
  losts.extend([x for x in range(12400,13000)])
  losts.extend([x for x in range(13300,13630)])
  losts.extend([x for x in range(13950,14053)])
  losts.extend([x for x in range(14250,14349)])
  return np.array(losts)

def train_filter(indices):
    valid = np.array([x not in get_lost_indices() for x in indices])
    return (indices == indices) & valid

def validation_filter(indices):
    valid = np.array([x not in get_lost_indices() for x in indices])
    return (indices == indices) & valid

def test_filter(indices):
    valid = np.array([x not in get_lost_indices() for x in indices])
    return (indices == indices) & valid

def anchor_filter(indices):
    valid = np.array([x not in get_lost_indices() for x in indices])
    return (indices == indices) & valid

def non_anchor_filter(indices):
    valid = np.array([x not in get_lost_indices() for x in indices])
    return (indices == indices) & valid