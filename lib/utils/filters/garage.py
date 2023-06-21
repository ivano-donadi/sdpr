def train_filter(indices):
  return (indices // 376 != 1) & (indices // 376 != 4)

def validation_filter(indices):
  return (indices // 376 == 1) | (indices // 376 == 4)

def test_filter(indices):
  return indices == indices

def anchor_filter(indices):
    return indices % 5 == 0

def non_anchor_filter(indices):
    return indices % 5 != 0