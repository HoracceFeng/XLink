from __future__ import print_function
import os 
import cv2
import sys
import argparse
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon
import cPickle as cpkl


'''
Annotation and Detection Load In
format:
    annot_file / dects_file    imgname, confidence, x1,y1 ,x2,y2, x3,y3, x4,y4
'''


def get_annotations(annot_filepath):
    print('>>> Start get_annotations() ...... \t', annot_filepath)
    true_dict = defaultdict(list)
    gt_pos = 0
    with open(annot_filepath, "r") as gt_file:
        for line in gt_file.readlines():
            nameID, confidence, _pts = line.split("\t")
            pts = eval(_pts)
            points = [[float(x), float(y)] for x, y in zip(pts[0::2], pts[1::2])]
            true_dict[nameID].append([confidence, points])
            gt_pos += 1
    return [true_dict, gt_pos]


def get_detections(dects_filepath):
    print('>>> Start get_detections() ...... \t', dects_filepath)
    pred_dict = defaultdict(list)
    pr_pos = 0
    with open(dects_filepath, "r") as pred_file:
        for line in pred_file.readlines():
            nameID, confidence, _pts = line.split("\t")
            pts = eval(_pts)
            # print("debug:", type(pts), pts)
            points = [[float(x), float(y)] for x, y in zip(pts[0::2], pts[1::2])]
            pred_dict[nameID].append([confidence, points])
            pr_pos += 1    
    return [pred_dict, pr_pos]


'''
Utilities
'''

def confidence_filter(_dict, confidence_thresh):
    num_box = 0
    if confidence_thresh == None:
        return _dict
    fixdict = {}
    for nameID in _dict.keys():
        fixdict[nameID] = []
        for info in _dict[nameID]:
            confidence, points = info
            if float(confidence) >= confidence_thresh:
                fixdict[nameID].append(points)
                num_box += 1
    return fixdict, num_box


'''
Evaluation Module
'''
def evaluate(annot_load, dects_load, annot_filter=None, confidence_thresh=None, iou_thresh=0.5):
    '''
    The logic has something wrong, see below.
    TODO: add _assignment flag to skip matched target
    '''
    print('Start Evaluations ......')
    true_dict, gt_pos = annot_load
    pred_dict, pr_pos = dects_load

    true_dict, gt_pos = confidence_filter(true_dict, annot_filter)
    pred_dict, pr_pos = confidence_filter(pred_dict, confidence_thresh)

    tp = 0
    img_list = list(true_dict.keys())
    img_list.sort()
    for nameID in img_list:
        # print("Processing image: " + nameID)
        # print("  >>> true_dict", true_dict[nameID])
        # print("  >>> pred_dict", pred_dict[nameID])

        ### filter for illegal box
        if not nameID in pred_dict.keys():
            continue

        # _true_dict = []
        # for true_obj in true_dict[nameID]:
        #     if len(true_obj) < 3:
        #         pr_pos -= 1
        #         continue
        #     else:
        #         _true_dict.append(true_obj)
        # if len(_true_dict) == 0:
        #     contionue

        pred_polys = [Polygon(pred_obj) for pred_obj in pred_dict[nameID]]
        true_polys = [Polygon(true_obj) for true_obj in true_dict[nameID]]

        '''
        Warning:    the logic of pred/true has something wrong.
                    since no _assignment flag to annotate those true poly that hit the pred poly
                    so the recall and precision will be over-estimated, so do F-score.
        '''
        for true_poly in true_polys:
            for pred_poly in pred_polys:
                iou = 0
                try:
                    inter = true_poly.intersection(pred_poly).area
                    union = true_poly.area + pred_poly.area - inter
                    iou = inter / union
                except:
                    print("Topological Error! IOU set to 0")
                    iou = 0
                if iou > iou_thresh:
                    tp += 1
                    break
            
    p = 1.0 * tp / pr_pos
    r = 1.0 * tp / gt_pos
    f = 2.0 * p * r / (p + r)

    print("#GTbox:{}\t#Dectbox:{}\t#hits:{}".format(gt_pos, pr_pos, tp))
    print("Precision: \t %.3f" % p)
    print("Recall: \t %.3f" % r)
    print("F-Score:  \t %.3f" % f)




if __name__ == "__main__":

    '''
    This Evaluation only support rectangle object right not
    Still not support evaluation with different class
    will update soon, in `MLT competition`
    '''

    ### Arguments
    parser = argparse.ArgumentParser(description='Polygon Evaluation Module -- For precision, recall and mAP calculation')
    parser.add_argument('--outdir',          default=None,  type=str,    help='output store in this document')
    parser.add_argument('--buffer',          default=None,  type=str,    help='buffer pkl file contains the annot and dect dictionary')
    parser.add_argument('--annotation_file', default=None,  type=str,    help='ground truth box text file (annot_file), skip if have buffer')
    parser.add_argument('--detection_file',  default=None,  type=str,    help='detection box output text file (dects_file), skip if have buffer')
    parser.add_argument('--annot_filter',    default=1.0,   type=float,  help='Annotations filter, annot_conf: <0 is curve, >= 0 is rectangle difficult, >0 is rectangle target')
    parser.add_argument('--confidence',      default=0.0,   type=float,  help='Confidence Threshold')
    parser.add_argument('--iou',             default=0.5,   type=float, help='IOU Threshold')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    ### Buffer load or write
    if os.path.exists(os.path.join(args.outdir, args.buffer)):
        _buffer                      = cpkl.load(open(os.path.join(args.outdir, args.buffer), 'rb'))
        annot_load                   = _buffer['annotations']
        dects_load                   = _buffer['detections']
        print("Pickle file load successfully \t", os.path.join(args.outdir, args.buffer))
    else:
        annot_load                   = get_annotations(args.annotation_file)
        dects_load                   = get_detections(args.detection_file)
        buffer_writer                = {}
        buffer_writer['annotations'] = annot_load
        buffer_writer['detections']  = dects_load
        cpkl.dump(buffer_writer, open(os.path.join(args.outdir, args.buffer), 'wb'))
        print("Pickle file Not be detected, auto-create success ...... \t", os.path.join(args.outdir, args.buffer))

    ### Parameters Display
    print( \
        'Evaluation Script for polygon object detection: \n \
         to change the parameters, try `python polygon_eval.py --help` \n \
            Parameters: \n \
            outdir:             {} \n \
            buffer:             {} \n \
            annotation_file:    {} \n \
            detection_file:     {} \n \
            annot_filter:       {} \n \
            confidence:         {} \n \
            iou:                {} \n'.format(args.outdir, args.buffer, args.annotation_file, args.detection_file, args.annot_filter, args.confidence, args.iou) 
    )

    ### Evaluation module
    evaluate(annot_load, dects_load, annot_filter=args.annot_filter, confidence_thresh=args.confidence, iou_thresh=args.iou)











