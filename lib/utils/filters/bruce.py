import numpy as np

def get_lost_indices():
  banned_indices = []
  banned_indices.append(3)
  banned_indices.extend(range(38, 52))
  banned_indices.extend(range(62,64))
  banned_indices.extend(range(65,72))
  banned_indices.extend(range(74,93))
  banned_indices.extend(range(106,107))
  banned_indices.extend(range(113,129))
  banned_indices.extend(range(130,132))
  banned_indices.extend(range(133,136))
  banned_indices.extend(range(148,174))
  banned_indices.extend(range(176,177))
  banned_indices.extend(range(179,185))
  return banned_indices

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