#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def create_hist(onedarray,title,bins=20):
    plt.hist(onedarray,bins=bins)
    if title:
        plt.title(title)
    else:
        plt.title('Histogramm')
    plt.xlabel('Wert')
    plt.ylabel('Häufigkeit')
    _fn = 'plots/%s.png' % title.replace(' ','_')
    plt.savefig(_fn)


def normpop(nparray):
    # normalisiert Häufigkeitswerte und wendet 1-(1-x)¹⁰
    pop_tmp = nparray / nparray.sum()
    pop_tmp = 1 - (1 - pop_tmp) ** 10
    print('Max:\t%s' % pop_tmp.max())
    print('Min:\t%s' % pop_tmp.min())
    print('Sum:\t%s' % pop_tmp.sum())
    return pop_tmp


def winkeldist(winkel0, winkel1, ret_ind=False):
    # winkel0:  1 Winkel
    # winkel1,  n Winkel

    if isinstance(winkel1, int) or isinstance(winkel1, np.float32):
        winkel1 = np.array(winkel1).reshape((1, -1))

    winkel_tmp = np.zeros((len(winkel1), 3), dtype='float32')

    print('winkel0: %s' % winkel0)
    print('winkel1: %s' % winkel1)

    winkel_tmp[:,0] = np.abs(winkel0 - winkel1)     # 0
    winkel_tmp[:,1] = (360 - winkel1) + winkel0     # 1
    winkel_tmp[:,2] = (360 - winkel0) + winkel1     # 2

    w_ind = winkel_tmp.argmin(axis=1)

    winkel_min = np.min(winkel_tmp, axis=1)
    if not ret_ind:
        return winkel_min
    else:
        return w_ind


def wmittel(nparray):
    # nparray enthält 2 Winkel, von denen die Mitte berechnet werden muss.

    w_dist = winkeldist(nparray[0], nparray[1])
    w_art = winkeldist(nparray[0], nparray[1], ret_ind=True)

    if w_art == 0:
        return np.mean(nparray)
    if w_art == 1:
        return (nparray[1] + w_dist/2) % 360
    if w_art == 2:
        return (nparray[0] + w_dist/2) % 360



if __name__ == '__main__':
    a = np.random.randint(0,360)
    b = np.random.randint(0,360,4)

    c = winkeldist(a, b, ret_ind=True)
    print('a: %s' % a)
    print('b:')
    print(b)
    print('c')
    print(c)