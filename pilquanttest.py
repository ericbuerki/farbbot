#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import cv2
from PIL import Image

bild_cv = cv2.imread('samples/bild02.jpg')
bild_np = cv2.cvtColor(bild_cv, cv2.COLOR_BGR2RGB)

print('bild_np.shape')
print(bild_np.shape)

#Nehme an, dass anz pix gerade
anz_pix_halbe = (bild_np.shape[0]*bild_np.shape[1])//2

farben = bild_np.reshape((2,anz_pix_halbe,3))
print('Anzahl Farben:\t%s' % farben.shape[1])

img = Image.fromarray(farben)

#farben_pil = img.getcolors(maxcolors=255**3)
#print(farben_pil)

palette = img.convert('P',palette=Image.ADAPTIVE,
                             colors=64)
palette = palette.convert('RGB')
print('Palette')
print(palette.getcolors())
