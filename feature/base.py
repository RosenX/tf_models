from collections import namedtuple

CategoryFeature = namedtuple('Feature', ['name', 'is_identity', 'bucket_size', 'default', 'dtype', 'emb_dim'])
DenseFeature = namedtuple('Feature', ['name', 'bucket_size', 'default', 'dtype'])