#!/usr/bin/env python3

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


def winkeldist(winkel0, winkel1):
    winkel_a = np.abs(winkel0 - winkel1)

    if winkel0 < winkel1:
        winkel_b = (360 - winkel1) + winkel0
    else:
        winkel_b = (360 - winkel0) + winkel1

    if winkel_a > winkel_b:
        return winkel_b
    else:
        return winkel_as


if __name__ == '__main__':
