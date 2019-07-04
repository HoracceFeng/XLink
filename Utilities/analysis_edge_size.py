import os
import numpy as np
import sys


def round_subtract_find_min_edge(ps):
    _out = []
    for i in range(len(ps)):
        if i == len(ps)-1:
            continue
        for j in range(len(ps)-i-1):
            _out.append(np.abs(ps[j]-ps[i]))
    return sorted(_out)[-2]


def get_xy_list(txtdir):
    txtlist = os.listdir(txtdir)
    minxlist = []
    minylist = []
    for txtfile in txtlist:
        _file = open(os.path.join(txtdir, txtfile)).readlines()
        for line in _file:
            x1,y1, x2,y2, x3,y3, x4,y4 = line.strip().split(',')[:8]
            x1=int(x1); y1=int(y1)
            x2=int(x2); y2=int(y2)
            x3=int(x3); y3=int(y3)
            x4=int(x4); y4=int(y4)
            pxs = [x1,x2,x3,x4]
            pys = [y1,y2,y3,y4]
            minxlist.append(round_subtract_find_min_edge(pxs))
            minylist.append(round_subtract_find_min_edge(pys))
    return minxlist, minylist


def main(minxlist, minylist, limit_ratio = 0.05, start_min_edge=5, start_min_area=100, printable=False):
    xratio = 0
    xlimit = 0
    while xratio < 0.05:
        xlimit += 5
        xratio = len([i for i in minxlist if i < xlimit])/float(len(minxlist))
    if printable:
        print "X minedge: ", xlimit, '\t', 'Num:', len([i for i in minxlist if i < xlimit])  

    yratio = 0
    ylimit = 0
    while yratio < 0.05:
        ylimit += 5
        yratio = len([i for i in minylist if i < ylimit])/float(len(minylist))
    if printable:
        print "Y minedge: ", ylimit, '\t', 'Num:', len([i for i in minylist if i < ylimit])

    minedge = min(xlimit, ylimit)
    minarea = xlimit*ylimit
    if minarea < start_min_area: 
        minarea = start_min_area

    return minedge, minarea



if __name__ == '__main__':

    txtdir = sys.argv[1]

    minxlist, minylist = get_xy_list(txtdir)

    minedge, minarea = main(minxlist, minylist, printable=True)

    print "Set Min EDGE As:", minedge
    print "Set Min AREA As:", minarea