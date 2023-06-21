def train_filter(indices):
  return indices == indices

def validation_filter(indices):
  return indices == indices

def test_filter(indices):
  return indices == indices

def anchor_filter(indices):
    return indices % 6 == 0

def non_anchor_filter(indices):
    return indices % 6 != 0
