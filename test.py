#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

from sklearn.cluster import DBSCAN, AffinityPropagation

from farben import VibrantPy


fn = 'samples/bild07.jpg'
vibr = VibrantPy(fn, r=200)

vibrant = vibr.farben_tot
print('vibrant.shape')
print(vibrant.shape)


db = DBSCAN(eps=10, min_samples=2).fit(vibrant[:,7:])

for i in np.unique(db.labels_):
    print('Label:\t%s' % i)
    print(vibrant[db.labels_==i])

print('db.core_sample_indices_')
print(db.core_sample_indices_)


'''
af = AffinityPropagation().fit(vibrant[:,7:])

print('af.cluster_centers_indices_')
print(af.cluster_centers_indices_)
print('Zentren')
print(vibrant[af.cluster_centers_indices_])
'''
