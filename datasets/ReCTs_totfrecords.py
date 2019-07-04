#encoding=utf-8
import numpy as np
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
import config
import json


def check_out_boundary(x, _min, _max):
    if _min < x < _max:
        x = x
    elif x <= _min:
        x = _min+1
    elif x >= _max:
        x = _max-1
    return x


def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print "%d images found in %s"%(len(image_names), data_path);
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            path = util.io.join_path(data_path, image_name);
            print "\tconverting image: %d/%d %s"%(idx, len(image_names), image_name);
            image_data = tf.gfile.FastGFile(path, 'r').read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];

            ## Annotations
            gt_name = image_name + '.json';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            # print 'gt_name', gt_name, gt_filepath

            ### json file 
            with open(gt_filepath) as jsonbuffer:
                jsondict = json.loads(jsonbuffer.read())

            print " >>>>>> ", image_name, w, jsondict['imageWidth'], h, jsondict['imageHeight']

            for obj in jsondict['shapes']:
                gt = []
                for pt in obj['points']:
                    ptx, pty = pt
                    ptx = check_out_boundary(ptx, 0, w)
                    pty = check_out_boundary(pty, 0 ,h)
                    gt.append(ptx)
                    gt.append(pty)
                num_pts = len(obj['points'])
                oriented_box = [int(gt[i]) for i in range(len(gt))]
                oriented_box = np.asarray(oriented_box) / ([w, h] * num_pts)
                oriented_bboxes.append(oriented_box)
                
                xs = oriented_box.reshape(num_pts, 2)[:, 0]
                ys = oriented_box.reshape(num_pts, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                print "    >>>>> ", xmin,ymin,xmax,ymax

                labels_text.append(str(obj['label'].encode('utf8')))  ## unicode to byte, should convert back to unicode in result 
                ignored = util.str.contains(obj['label'], '#')
                if ignored:
                    labels.append(config.ignore_label);
                else:
                    labels.append(config.text_label)
            example = convert_to_example(image_data, image_name, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":

    ### ICDAR-ReCTs-2019
    root_dir = util.io.get_absolute_path('/data/ICDAR2019/ReCTS/data')
    output_dir = util.io.get_absolute_path('/data/ICDAR2019/ReCTS/data/TFRecords/')
    _sets = 'train'

    util.io.mkdir(output_dir);
    training_data_dir = util.io.join_path(root_dir, 'JPEGImages', _sets)
    training_gt_dir = util.io.join_path(root_dir,'Annotations', _sets)
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'ICDAR2019-ReCTs_train.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)

