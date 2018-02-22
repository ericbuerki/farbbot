#!/usr/bin/env python3

import os

import cv2
import numpy as np

import svgwrite
from cairosvg import svg2png


class Bild(object):
    def __init__(self, filename, bild, farben, debug=False, pop=False):

        self.bild = bild            #Bild (als Numpy-Array)
        self.farben = farben        #Dominante Farben (als Numpy-Array)
        self.debug = debug          #Boolean für Debugfunktionen
        self.pop = pop

        self.fn_pal_svg = self.dateiname(filename,
                                        '_palette.svg',
                                        bn=True,
                                        ordner='paletten')  #Palette (svg)
        self.fn_pal_png = self.dateiname(filename,
                                        '_palette.png',
                                        bn=True,
                                        ordner='paletten')  #Palette (png)
        self.fn_png = self.dateiname(filename,
                                     '_merged.png',
                                     bn=True,
                                     ordner='fertig')       #Bild+Palette


        if not self.debug:
            self.h, self.w = bild.shape[:2]
        else:
            self.w = 1920
            self.h = 1080
            self.fn_pal_svg = self.dateiname(filename,
                                        '_palette.svg',
                                        bn=True,
                                        ordner='paletten')  #Palette (svg)

        self.hp = self.h * 0.25

        if self.hp >= self.w/len(self.farben):
            self.hp *= 0.6


    def erstelle_palette(self):
        palette = svgwrite.Drawing(filename = self.fn_pal_svg,
                                   size = (str(self.w) + 'px', str(self.hp) + 'px'))

        breite = self.w/len(self.farben)+1
        abstand_y = self.hp * 0.8
        '''
        font_size = breite * 0.2
        if self.farben.shape[0] < 5 or len(self.farben.shape) == 1:
            font_size = breite * 0.1
        '''
        
        font_size = 64
        
        text_style = 'font-size:%ipx; font-family:%s' % (font_size,
                     'Acumin Pro SemiCondensed')

        for i in range(len(self.farben)):
            farbe = 'rgb(%s)' % self.farbe2string(self.farben[i])
            xpos = i/len(self.farben)*self.w
            #print(xpos)

            #Zeichne Palette
            palette.add(palette.rect(insert = (xpos-1, 0),
                size = (str(breite)+'px', (str(self.hp+1))+'px'),
                fill = farbe))


            #Zeichne Beschriftung
            beschriftung = self.rgb_to_hex(self.farben[i])
            abstand_x = xpos + (breite * 0.1)
            if self.luma(self.farben[i]) < 80:
                color = 'white'
            else:
                color = 'black'

            text = palette.text(beschriftung, insert=(abstand_x, abstand_y),
                                            fill=color, style=text_style)
            if len(self.farben) < 20:
                palette.add(text)

            if self.debug:
                abstand_y_db = self.hp*0.3
                font_size_db = breite*0.2
                if self.farben.shape[0] < 5 or len(self.farben.shape) == 1:
                    font_size_db = breite*0.1

                font_size_db = 64
                
                try:
                    strpop = '%.2f' % self.pop[i]
                except:
                    try:
                        strpop = '%.1f/%.1f/%.1f' % (self.pop[i][0],
                                                     self.pop[i][1],
                                                     self.pop[i][2])
                    except:
                        strpop = 'n/a'

                text_style_db = 'font-size:%ipx; font-family:%s' % (font_size_db,
                                            'Acumin Pro SemiCondensed')
                text_db = palette.text(strpop,
                                       insert=(abstand_x, abstand_y_db),
                                       fill=color, style=text_style_db)
                if len(self.farben) < 20:
                    palette.add(text_db)

        palette.save()


    # Fügt Bild und Palette zusammen
    def merge(self):
        print('Füge Bild zusammen')
        svg2png(url=self.fn_pal_svg, write_to=self.fn_pal_png)
        palette_png = cv2.imread(self.fn_pal_png)

        print('self.bild.shape: %s,%s,%s' % self.bild.shape)
        print('palette_png.shape: %s,%s,%s' % palette_png.shape)

        h_bild = self.bild.shape[0] - palette_png.shape[0]

        bild_merged = np.concatenate((self.bild[:h_bild,:], palette_png),
                                     axis=0)
        cv2.imwrite(self.fn_png, bild_merged)
        if not self.debug:
            os.system('rm %s %s' % (self.fn_pal_svg,
                                    self.fn_pal_png))
        print('Fertig')

    # Hilfsfunktion für die Benennung der Dateien
    def dateiname(self, text, endung=False, bn=False, ordner=''):
        name = os.path.splitext(text)[0]
        if endung:
            name += endung
        if bn:
            name = os.path.basename(name)
            name = ordner + '/' + name
        return name

    # Gibt Numpy-Array als (svgwrite-kompatiblen) String aus
    def farbe2string(self, farbe):
        r, g, b = farbe
        string = '%s,%s,%s' % (r, g, b)
        return string

    # Konvertiert RGB (np.array-Item) zu Hex (string)
    def rgb_to_hex(self, rgb):
        rgb = tuple(rgb)
        return '#%02x%02x%02x' % rgb

    # Gibt Luma von Farbe (RGB) aus
    def luma(self, farbe):
        luma = (farbe[0] * 2 + farbe[1] * 3 + farbe[2])/6
        return luma

    # Gibt Pfad des zusammengefügten Bilds aus
    # Kann erst nach merge() ausgeführt werden
    def get_fn(self):
        return self.fn_png

    # Gibt Pfad der Palette (SVG) aus
    # Kann erst nach erstelle_palette() ausgeführt werden
    def get_fn_pal(self):
        return self.fn_pal_svg
