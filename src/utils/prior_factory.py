import numpy as np
from math import sin,cos,sqrt

def uniform(batch_size, n_dim, n_labels=10, minv=-1, maxv=1, label_indices=None):
    if label_indices is not None:
        if n_dim != 2 or n_labels != 10:
            raise Exception("n_dim must be 2 and n_labels must be 10.")

        def sample(label, n_labels):
            num = int(np.ceil(np.sqrt(n_labels)))
            size = (maxv-minv)*1.0/num
            x, y = np.random.uniform(-size/2, size/2, (2,))
            i = label / num
            j = label % num
            x += j*size+minv+0.5*size
            y += i*size+minv+0.5*size
            return np.array([x, y]).reshape((2,))

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
    else:
        z = np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)
    return z

def normal(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
  if use_label_info:
    if n_dim != 2 or n_labels != 10:
      raise Exception("n_dim must be 2 and n_labels must be 10.")

    def sample(n_labels):
      x, y = np.random.normal(mean, var, (2,))
      angle = np.angle((x-mean) + 1j*(y-mean), deg=True)
      dist = np.sqrt((x-mean)**2+(y-mean)**2)

      # label 0
      if dist <1.0:
        label = 0
      else:
        label = ((int)((n_labels-1)*angle))//360

      if label<0:
        label+=n_labels-1

      label += 1

      return np.array([x, y]).reshape((2,)), label

    z = np.empty((batch_size, n_dim), dtype=np.float32)
    z_id = np.empty((batch_size), dtype=np.int32)
    for batch in range(batch_size):
      for zi in range((int)(n_dim/2)):
        a_sample, a_label = sample(n_labels)
        z[batch, zi*2:zi*2+2] = a_sample
        z_id[batch] = a_label
    return z, z_id
  else:
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z

def hypersphere(batch_size, n_dim, mean=0, var=0.5):
  z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
  d = np.random.normal(mean, var, (n_dim, batch_size)).astype(np.float32)
  r = np.sqrt((d**2 + 1e-8).sum(axis=0))
  d = np.rollaxis((d / r), 1, 0)
  z = z + d
  return z

def hypersphere2(batch_size, w, n_dim, mean=0, var=0.5):
  z = np.random.normal(mean, var, (batch_size, w, n_dim)).astype(np.float32)
  d = np.random.normal(mean, var, (n_dim, batch_size, w)).astype(np.float32)
  r = np.sqrt((d**2 + 1e-8).sum(axis=0))
  d = np.rollaxis((d / r), 0, 2)
  z = z + d
  return z
