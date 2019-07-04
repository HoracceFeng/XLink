from __future__ import print_function 
import os
import sys
import json


'''
Clockwise direction check and fix
'''
from complex_revert_four_points import fourpoint


'''
Utilities
'''
def valid_list(_list, maxlen):
    fixed = []
    for i in _list:
        if i < 0:
            i = 0
        if i > maxlen:
            i = maxlen
        fixed.append(i)
    return fixed


'''
Decoder Definition to Parse different Annotation file 
confidence in gtbox: 
    if not rectangle:
        confidence = -1
        if ignore=1: 
            confidence = 0
        else:
            confidence=1
'''

def RECT_json(jsonpath, ignore_char=True):
    bboxes = []
    with open(jsonpath) as jsonbuffer:
        jsondict = json.loads(jsonbuffer.read())

    for kindid, kind in enumerate(jsondict.keys()):
        if ignore_char:
            if kind == 'char':
                continue
        for objid, obj in enumerate(jsondict[kind]):
            # x1,y1, x2,y2, x3,y3, x4,y4 = obj['points']
            confidence = 1-int(float(obj['ignore']))
            if len(obj['points']) != 8:
                confidence = -1
            if confidence > 0:
                if text.find('#') != -1:
                    confidence = 0
                else:
                    confidence = 1
            bboxes.append([confidence, obj['points']])

    return bboxes


def RCTW_json(jsonpath):
    bboxes = []
    with open(jsonpath) as jsonbuffer:
        jsondict = json.loads(jsonbuffer.read())

    for objid, obj in enumerate(jsondict['shapes']):
        pts = []; confidence = 1
        for pt in obj['points']:
            xi, yi = pt
            pts.append(xi)
            pts.append(yi)
        text = obj['label']
        if len(pts) != 8:
            confidence = -1
        if confidence > 0:
            if text.find('#') != -1:
                confidence = 0
            else:
                confidence = 1
        bboxes.append([confidence, pts])

    return bboxes


def txtparser(txtpath, setname='MLT'):
    txtdict = []
    bboxes = []
    txtlines =  open(txtpath).readlines()
    for line in txtlines:
        txtdict.append(line.strip().split(','))    

    for obj in txtdict:
        if setname == 'MLT':
            x1,y1, x2,y2, x3,y3, x4,y4, lang, text = obj
        elif setname == 'IST':
            x1,y1, x2,y2, x3,y3, x4,y4, text = obj
        else:
            print("*** FORMAT Error: format other than MLT/IST set should be redefined in code")
            pass
        pts = [int(x1.replace('\xef\xbb\xbf', '')), int(y1), int(x2),int(y2), int(x3),int(y3), int(x4),int(y4)]
        if text.find('#') != -1:
            confidence = 0
        else:
            confidence = 1
        bboxes.append([confidence, pts])

    return bboxes




if __name__ == '__main__':

    annotation_dir = sys.argv[1]
    decoder_type   = sys.argv[2]  ## 'either RCTW_json' or 'RECT_json' or 'txt'
    outfiledir     = sys.argv[3]

    annotlist = os.listdir(annotation_dir)
    if os.path.exists(outfiledir):
        print("Error: ", outfiledir, "is exists, please give another name")
        exit()
    outfile = open(outfiledir, 'w')

    for annotname in annotlist:
        nameID = annotname.split('.')[0]
        annotpath = os.path.join(annotation_dir, annotname)
        print(annotpath)
        if decoder_type == 'RCTW_json':
            bboxes = RCTW_json(annotpath)
        elif decoder_type == 'RECT_json':
            bboxes = RECT_json(annotpath)
        elif decoder_type == 'txt':
            bboxes = txtparser(annotpath)
        for info in bboxes:
            confidence, bbox = info
            if len(bbox) == 8:
                (x1,y1), (x2,y2), (x3,y3), (x4,y4)  = fourpoint(bbox)
                bbox = [x1,y1, x2,y2, x3,y3, x4,y4]
            print('>>> {}\t{}\t{}\n'.format(nameID, confidence, bbox))
            outfile.write('{}\t{}\t{}\n'.format(nameID, confidence, bbox))

    outfile.close()














