#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image

from skimage import color
from sklearn.cluster import DBSCAN, AffinityPropagation


from palette import Bild

np.set_printoptions(precision=1, edgeitems=7, suppress=True)
farbnamen = ['Vibrant', 'Muted', 'DarkVibrant',
             'DarkMuted', 'LightVibrant', 'LightMuted']

class VibrantPy(object):
    def __init__(self, filename, r=200, k=64):

        bild = Image.open(filename)

        if r:
            bild.thumbnail((r,r))

        farben_raw = bild.getcolors(maxcolors=255**3)
        farben_tmp = np.zeros((len(farben_raw), 10), dtype='float32')

        # 0:    Hue
        # 1:    Saturation
        # 2:    Value
        # 3:    Population
        # 4:    R
        # 5:    G
        # 6:    B
        # 7:    L
        # 8:    a
        # 9:    b

        row = 0
        for farbe in farben_raw:
            farben_tmp[row][3] = farbe[0]
            farben_tmp[row][4:7] = farbe[1]
            row += 1
        farben_tmp[:,3] /= np.max(farben_tmp[:,3])
        farben_tmp[:,3] *= 255

        alle = Farben(farben_tmp, 6)
        self.farben_tot = alle.get_farben()
        self.farben = []

        # 0-2:  *Vibrant
        # 3-5:  *Muted
        self.farben_len = np.empty(6, dtype='uint8')
        for i in range(3):      #Vibrant
            self.farben.append(Farben(self.farben_tot, (i*2)))
            self.farben_len[i] = len(self.farben[i])
        for i in range(3):
            self.farben.append(Farben(self.farben_tot, i*2+1))
            self.farben_len[i+3] = len(self.farben[i+3])

        v_sort = self.farben_len[:3].argsort()
        m_sort = self.farben_len[3:].argsort()+3
        sort = np.hstack((v_sort, m_sort))
        print('farben_len:\t%s' % self.farben_len)
        print('sort:\t\t%s' % sort)

        self.farben[v_sort[1]].deldup(self.farben[v_sort[0]].get_farben())
        self.farben[v_sort[2]].deldup(self.farben[v_sort[1]].get_farben())
        self.farben[m_sort[1]].deldup(self.farben[m_sort[0]].get_farben())
        self.farben[m_sort[2]].deldup(self.farben[m_sort[1]].get_farben())


        for farbe in self.farben:
            farbe.target()
        
        '''
        for farbe in self.farben:
            #print(farbe.get_mode())
            if len(farbe) > 1:
                farbe.cluster('db_hue')
            else:
                print('len(farbe) < 1')
                print(farbe.get_farben())

        for i in range(3):      #Vibrant
            self.farben_len[i] = len(self.farben[i])
        for i in range(3):
            self.farben_len[i+3] = len(self.farben[i+3])

        v_sort = self.farben_len[:3].argsort()
        m_sort = self.farben_len[3:].argsort()
        '''



        order = [0, 3, 1, 4, 2, 5]
        self.farben = [self.farben[i] for i in order]

        self.farben_list = []
        for i in range(6):
            self.farben_list.append(self.farben[i].get_farben())


    def get_farben(self):
        return self.farben_list

    def get_farben_tot(self):
        return self.farben_tot



class Farben(object):
    def __init__(self, farben, modus=6, rec=False):
        self.modus = modus
        self.farben = farben
        self.rec = rec      # ob rekursiv oder nicht

        # 0-5: Farbnamen, 6: Alle, 7: (beinahe) Fertig (für self.target(delta))
        if self.modus < 6:
            self.select()
            #self.cluster()
            #self.target()
        if self.modus == 6:
            self.recomp('rgb',['hsv','lab'])
            self.cluster('init')
            #self.quantize(k=128)
        if self.rec:
            self.target()


    def __len__(self):
        return self.farben.shape[0]

    def __lt__(self, other):
        return self.farben.shape[0] < other

    def __le__(self, other):
        return self.farben.shape[0] <= other

    def __eq__(self, other):
        return self.farben.shape[0] == other

    def __ne__(self, other):
        return self.farben.shape[0] != other

    def __gt__(self, other):
        return self.farben.shape[0] > other

    def __ge__(self, other):
        return self.farben.shape[0] >= other

    def __bool__(self):
        # True, falls Farben in Objekt, ansonsten False
        if 0 in self.farben.shape:
            return True
        else:
            return False

    def recomp(self, orig, dest):
        # orig: Ausgangsfarbraum
        # dest: Zielfarbraum/räume
        #print('dest: %s' % dest)
        if len(self.farben.shape) == 2:                 # Mehrere Farben
            if orig == 'hsv':
                f_temp = np.copy(self.farben[:,:3])
                f_temp[:,0] /= 360              # 179
                f_temp[:,1:] /= 255
                if 'rgb' in dest:
                    f_temp_rgb = color.hsv2rgb([f_temp])[0]
                    f_temp_rgb *= 255
                    self.farben[:,4:7] = f_temp_rgb
                if 'lab' in dest:
                    f_temp_hsv = color.hsv2rgb([f_temp])
                    f_temp_hsv = color.rgb2lab(f_temp_hsv)[0]
                    self.farben[:,7:] = f_temp_hsv
            if orig == 'rgb':
                f_temp = np.copy(self.farben[:,4:7])
                f_temp /= 255
                if 'hsv' in dest:
                    f_temp_hsv = color.rgb2hsv([f_temp])[0]
                    f_temp_hsv[:,0] *= 360      # 179
                    f_temp_hsv[:,1:] *= 255
                    self.farben[:,:3] = f_temp_hsv
                if 'lab' in dest:
                    f_temp_lab = color.rgb2lab([f_temp])[0]
                    self.farben[:,7:] = f_temp_lab
            if orig == 'lab':
                f_temp = np.copy(self.farben[:,7:])
                if 'hsv' in dest:
                    f_temp[:,0] /= 100
                    f_temp[:,1:] /= 128
                    f_temp_hsv = color.lab2rgb([f_temp])
                    f_temp_hsv = color.rgb2hsv(f_temp_hsv)[0]
                    f_temp_hsv[:,0] *= 360      # 179
                    f_temp_hsv[:,1:] * 255
                if 'rgb' in dest:
                    f_temp_rgb = color.lab2rgb([f_temp])[0]
                    f_temp_rgb *= 255
                    self.farben[:,4:7] = f_temp_rgb
        if len(self.farben.shape) == 1:                 # Einzelne Farbe
            if orig == 'hsv':
                f_temp = np.copy(self.farben[:3])
                f_temp[0] /= 360                # 179
                f_temp[1:] /= 255
                if 'rgb' in dest:
                    f_temp_rgb = color.hsv2rgb([f_temp])[0]
                    f_temp_rgb *= 255
                    self.farben[4:7] = f_temp_rgb
                if 'lab' in dest:
                    f_temp_hsv = color.hsv2rgb([f_temp])
                    f_temp_hsv = color.rgb2lab(f_temp_hsv)[0]
                    self.farben[7:] = f_temp_hsv
            if orig == 'rgb':
                f_temp = np.copy(self.farben[4:7])
                f_temp /= 255
                if 'hsv' in dest:
                    f_temp_hsv = color.rgb2hsv([f_temp])[0]
                    f_temp_hsv[0] *= 360        # 179
                    f_temp_hsv[1:] *= 255
                    self.farben[:3] = f_temp_hsv
                if 'lab' in dest:
                    f_temp_lab = color.rgb2lab([f_temp])[0]
                    self.farben[7:] = f_temp_lab
            if orig == 'lab':
                f_temp = np.copy(self.farben[7:])
                if 'hsv' in dest:
                    f_temp_hsv = color.lab2rgb([f_temp])
                    f_temp_hsv = color.rgb2hsv(f_temp_hsv)[0]
                    f_temp_hsv[0] *= 360        # 179
                    f_temp_hsv[1:] *= 255
                if 'rgb' in dest:
                    f_temp_rgb = color.lab2rgb([f_temp])[0]
                    f_temp_rgb *= 255
                    self.farben[4:7] = f_temp_rgb
                    
        if self.modus == 6:
            print('self.modus == %s' % self.modus)
            cond = np.logical_and(self.farben[:,2] > 25, self.farben[:,1] > 50)
            self.farben=self.farben[cond]

    def quantize(self, k=64):
        mode = 1
        # Quantisiert Bild in k Farben
        if (self.farben.shape[0] % 2) is not 0:
            print('self.farben.shape')
            print(self.farben.shape)
            print('Anzahl Farben ist ungerade, experimentell')
            self.farben = np.vstack((self.farben,
                                     np.zeros(10, dtype='float32')))

        f_temp = np.uint8(self.farben[:,4:7])
        f_temp = f_temp.reshape((2, self.farben.shape[0]//2, 3))
        b_temp = Image.fromarray(f_temp)
        if mode is 0:
            pal = b_temp.convert('P', palette=Image.ADAPTIVE,
                                 colors=k)
        if mode is 1:
            pal = b_temp.quantize(colors=k, method=2)

        pal = pal.convert('RGB')
        farben = pal.getcolors()
        self.farben = np.zeros((len(farben), 10), dtype='float32')
        row = 0
        for farbe in farben:
            self.farben[row][3] = farbe[0]
            self.farben[row][4:7] = farbe[1]
            row += 1

        self.farben[:,3] /= np.max(self.farben[:,3])
        self.farben[:,3] *= 255
        # Löscht farben mit Pop < 3
        self.farben = self.farben[self.farben[:,3] > 3]
        self.recomp('rgb', ['hsv', 'lab'])

    def select(self):
        # Wählt passende Farbe aus
        # Werte übernommen von vibrant.js

        max_dark_luma = 115             # 115
        min_light_luma = 141            # 141

        min_normal_luma = 77            # 77
        max_normal_luma =  179

        max_muted_saturation = 102
        min_vibrant_saturation = 90

        if self.modus == 0:          # Vibrant
            bedingung = np.logical_and(self.farben[:,2] > min_normal_luma,
                            self.farben[:,2] < max_normal_luma)
            bedingung = np.logical_and(bedingung,
                            self.farben[:,1] > min_vibrant_saturation)
        elif self.modus == 1:        # Muted
            bedingung = np.logical_and(self.farben[:,2] > min_normal_luma,
                            self.farben[:,2] < max_normal_luma)
            bedingung = np.logical_and(bedingung,
                            self.farben[:,1] < max_muted_saturation)
        elif self.modus == 2:        # DarkVibrant
            bedingung = np.logical_and(self.farben[:,2] < max_dark_luma,
                            self.farben[:,1] > min_vibrant_saturation)
        elif self.modus == 3:        # DarkMuted
            bedingung = np.logical_and(self.farben[:,2] < max_dark_luma,
                            self.farben[:,1] < max_muted_saturation)
        elif self.modus == 4:        # LightVibrant
            bedingung = np.logical_and(self.farben[:,2] > min_light_luma,
                            self.farben[:,1] > min_vibrant_saturation)
        elif self.modus == 5:        # LightMuted
            bedingung = np.logical_and(self.farben[:,2] > min_light_luma,
                            self.farben[:,1] < max_muted_saturation)

        self.farben = self.farben[bedingung]


        only_pop = self.farben[:,3] > 5
        if np.any(only_pop) == True:
            noise = (np.sum(~only_pop)/len(only_pop))*100
            #print('Rauschen: %.2f%%' % noise)
            self.farben = self.farben[only_pop]

    def cluster(self,mode='af'):
        if mode == 'af':
            af = AffinityPropagation().fit(self.farben[:,4:7])
            self.farben = self.farben[af.cluster_centers_indices_]

        if mode == 'db':
            db = DBSCAN(eps=40, min_samples=0).fit(self.farben[:,4:7])
            self.farben = self.farben[db.labels_]
            self.farben = np.unique(self.farben, axis=0)
            
        if mode == 'db_hue':
            hue_sin = np.sin((self.farben[:,0]/360)*2*np.pi)    # -> x
            hue_sin = hue_sin.reshape((-1,1))
            hue_cos = np.cos((self.farben[:,0]/360)*2*np.pi)    # -> y
            hue_cos = hue_cos.reshape((-1,1))
            
            
            db = DBSCAN(eps=0.1, min_samples=0)\
                 .fit(np.hstack((hue_sin, hue_cos)))
            noise = (len(db.labels_[db.labels_==-1])/len(db.labels_))*100
            print('\t\tRauschen: %.2f %%' % noise)

            if len(set(db.labels_)) > 2:
                labels = set(db.labels_[db.labels_ != -1])
            else:
                labels = set(db.labels_)
            farben_tmp = np.zeros((len(labels),10), dtype='float32')

            mode_select = 0          # 0: target, 1: np.mean
            for i in labels:
                #print('Farbe %s' % i)
                #print(self.farben[db.labels_ == i][:,4:7])
                if mode_select == 0:
                    f_tmp =  Farben(self.farben[db.labels_ == i],
                                    modus=self.modus, rec=True)
                    #print('\t\tf_tmp.get_farben()')
                    #print(f_tmp.get_farben())
                    farben_tmp[i] = f_tmp.get_farben()
                if mode_select == 1:
                    f_tmp =  np.mean(self.farben[db.labels_ == i][:,4:7],
                                     axis=0)
                    pop = self.farben[db.labels_ == i][:,3].sum()
                    farben_tmp[i,4:7] = f_tmp
                    farben_tmp[i,3] = pop
            #print('\t\tfarben_tmp')
            #print(farben_tmp)
            self.farben = farben_tmp
            #print('self.farben')
            #print(self.farben)
            self.recomp('rgb',['hsv','lab'])

        if mode=='init':
            print('clustere Bild,\tmodus=%s' % mode)
            hue_sin = np.sin((self.farben[:,0]/360)*2*np.pi)    # -> x
            hue_sin = hue_sin.reshape((-1,1))
            hue_cos = np.cos((self.farben[:,0]/360)*2*np.pi)    # -> y
            hue_cos = hue_cos.reshape((-1,1))
            farben_sv = np.copy(self.farben[:,1:3])/255   #255/127.5/63.75
            hsv_fitted = np.hstack((hue_sin, hue_cos, farben_sv))
            
            
            db = DBSCAN(eps=0.05, min_samples=0)\
                 .fit(hsv_fitted)

            noise = (len(db.labels_[db.labels_==-1])/len(db.labels_))*100
            print('\t\tRauschen: %.2f %%' % noise)

            labels = set(db.labels_[db.labels_ != -1])
            print('Anzahl labels: %s' % len(labels))
            farben_tmp = np.zeros((len(labels),10), dtype='float32')
            
            for i in labels:
                cond = self.farben[db.labels_==i][:,3].argmax()
                f_tmp = self.farben[db.labels_==i][cond]
                pop = self.farben[db.labels_ == i][:,3].sum()
                farben_tmp[i] = f_tmp
                farben_tmp[i,3] = pop
            self.farben = farben_tmp
            self.recomp('rgb',['hsv','lab'])
            self.farben = self.farben[self.farben[:,3].argsort()][::-1]

    def target(self, enable_delta=False):
        # Wählt passendste Farbe aus
        # Werte übernommen von vibrant.js
        target_dark_luma = 67
        target_normal_luma = 128
        target_light_luma = 189
        target_muted_saturation = 77
        target_vibrant_saturation = 256

        wl = 6      # Gewichtung des Lumas
        ws = 3      # Gewichtung der Sättigung
        wp = 1      # Gewichtung der Häufigkeit

        if enable_delta:
            wl = 1
            ws = 2
            wp = 6

        if self.rec:
            wl = 0
            ws = 1
            wp = 2

        if self.modus == 0:          # Vibrant
            target0 = np.abs(self.farben[:,2]-target_normal_luma)
            target1 = np.abs(self.farben[:,1]-target_vibrant_saturation)
        elif self.modus == 1:        # Muted
            target0 = np.abs(self.farben[:,2]-target_normal_luma)
            target1 = np.abs(self.farben[:,1]-target_muted_saturation)
        elif self.modus == 2:        # DarkVibrant
            target0 = np.abs(self.farben[:,2]-target_dark_luma)
            target1 = np.abs(self.farben[:,1]-target_vibrant_saturation)
        elif self.modus == 3:        # DarkMuted
            target0 = np.abs(self.farben[:,2]-target_dark_luma)
            target1 = np.abs(self.farben[:,1]-target_muted_saturation)
        elif self.modus == 4:        # LightVibrant
            target0 = np.abs(self.farben[:,2]-target_light_luma)
            target1 = np.abs(self.farben[:,1]-target_vibrant_saturation)
        elif self.modus == 5:        # LightMuted
            target0 = np.abs(self.farben[:,2]-target_light_luma)
            target1 = np.abs(self.farben[:,1]-target_muted_saturation)

        target2 = np.abs(self.farben[:,3]-255)

        delta = (target0 * wl + target1 * ws + target2 * wp)/3

        target = delta.argsort()
        target = target[0]
        if enable_delta:
            self.delta = delta
        else:
            self.farben = self.farben[target]

    def deldup(self, farben_comp):
        # Löscht Doppelte Farben im Array
        # Anwendung nach Anzahl Farben
        #   farben_comp->self.farben
        #   0 -> 1, 1 -> 2

        keep = np.ones(self.farben.shape[0], dtype='bool')

        for i in range(self.farben.shape[0]):
            for j in range(farben_comp.shape[0]):
                # print('i: %s   j: %s' % (i,j))
                if np.all(self.farben[i] == farben_comp[j]):
                    keep[i] = 0

        self.farben = self.farben[keep]

    def select_final(self, f_avoid=False, f_compare=None):
        # Wählt Farben für endgültige Palette aus
        self.target(enable_delta=True)
        print('self.delta')
        print(self.delta)
        print('self.farben')
        print(self.farben)
        print('f_compare')
        print(f_compare)
        if f_avoid == False:
            self.farben = self.farben[self.delta.argmin()]
            print('Ausgewählte Farbe in %s:' % farbnamen[self.modus])
            print(self.farben)
        else:
            anz_avoid = np.empty(self.farben.shape[0], dtype='uint8')
            count = 0
            for farbe in self.farben[:,0]:
                if len(f_compare.shape) == 2:
                    w_dist = self.winkel_dist(farbe, f_compare[:,0])
                if len(f_compare.shape) == 1:
                    w_dist = self.winkel_dist(farbe, f_compare[0])
                anz_avoid[count] = len(w_dist[w_dist < 20])
                count += 1

            print('anz_avoid')
            print(anz_avoid)
        print('='*80)

    def winkel_dist(self, winkel0, winkel1):
        a = winkel1 - winkel0
        b = winkel1 - (360 - winkel0)
        try:
            dist = np.min(np.hstack((a, b)), axis=0)
            print('winkel_dist: try')
        except:
            dist = np.hstack((a, b))
            print('winkel_dist: except')
        return dist

    def get_farben(self):
        return self.farben

    def isempty(self):
        if 0 in self.farben.shape:
            return True
        else:
            return False

    def get_mode(self):
        if self.modus < 6:
            return farbnamen[self.modus]
        if self.modus == 6:
            return 'alle'
        if self.modus == 7:
            return 'leer'

if __name__ == '__main__':
    os.system('rm paletten/*')
    fn = 'samples/bild12.jpg'
    # os.system('eog %s' % fn)
    vibrant = VibrantPy(fn, r=200)

    farben_tot = vibrant.get_farben_tot()
    farben_tot = farben_tot[farben_tot[:,3] > 1]
    farben_tot = farben_tot[farben_tot[:,0].argsort()]
    farben_list = vibrant.get_farben()

    f_count = 0
    for farben in farben_list:
        fn_temp = os.path.splitext(fn)
        fn_temp = '%s_%s_%s%s' % (fn_temp[0], f_count,
                                  farbnamen[f_count], fn_temp[1])
        fn_temp_np = 'farben/%s_%s.csv' % (f_count, farbnamen[f_count])
        np.savetxt(fn_temp_np, farben, delimiter=',')
        try:
            palette = Bild(fn_temp, None, np.uint8(farben[:,4:7]),
                           debug=True, pop=farben[:,3])
            palette.erstelle_palette()
        except:
            try:
                palette = Bild(fn_temp, None, np.uint8([farben[4:7]]),
                               debug=True, pop=farben[3])
                palette.erstelle_palette()
            except:
                print('Palette für %s konnte nicht generiert werden'
                      % farbnamen[f_count])
        f_count += 1

    palette = Bild(fn, None, np.uint8(farben_tot[:,4:7]),
                   debug=True, pop=farben_tot[:,3])
    palette.erstelle_palette()

    '''
    farben = farben[farben[:,7].argsort()[::-1]]
    print('farben')
    print(farben)
    np.savetxt('farben/vibrant.csv', farben, delimiter=',')
    print('farben.shape[0]')
    print(farben.shape[0])

    palette = Bild(fn, None, np.uint8(farben[:,4:7]),
                   debug=True, pop=farben[:,3])
    palette.erstelle_palette()
    '''
