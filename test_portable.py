#encoding = utf-8

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import metrics as tfe_metrics
import cv2, os
import pixel_link
from nets import pixel_link_symbol
import time

slim = tf.contrib.slim
import config


# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
  'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_dir', 'None', 
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'output_dir', 'None', 
    'The directory where the output image store.')
tf.app.flags.DEFINE_bool('draw_image', False, 'output result image or not')

# tf.app.flags.DEFINE_integer('eval_image_width', None, 'resized image width for inference')
# tf.app.flags.DEFINE_integer('eval_image_height',  None, 'resized image height for inference')
tf.app.flags.DEFINE_float('scale_resize', 1, 'resize image by ratio')
tf.app.flags.DEFINE_float('pixel_conf_threshold',  None, 'threshold on the pixel confidence')
tf.app.flags.DEFINE_float('link_conf_threshold',  None, 'threshold on the link confidence')


tf.app.flags.DEFINE_bool('using_moving_average', True, 
                         'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')


FLAGS = tf.app.flags.FLAGS


if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)


def config_initialization():
    # image shape and feature layers shape inference
    # image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    image_shape = (512,512)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = FLAGS.pixel_conf_threshold,
                       link_conf_threshold = FLAGS.link_conf_threshold,
                       num_gpus = 1, 
                   )


def scale_resize(image, scale):
    h,w,_ = image.shape
    resize_shape = (int(w*scale), int(h*scale))
    image = cv2.resize(image, resize_shape)
    return image


def revert_dectbox(_boxes, scale):
    '''
    revert scale back to origin, default `_boxes` should be np.array
    '''
    boxes = []
    for _box in _boxes:
        box = np.array(_box) / float(scale)
        box = box.astype('int')
        boxes.append(box)
    return boxes


def test():
    outfile = os.path.join(FLAGS.output_dir, 'DECT_result.txt')
    if os.path.exists(outfile):
        os.remove(outfile)
    wfile = open(outfile, 'w')
    # print ">> scale_resize", FLAGS.scale_resize, type(FLAGS.scale_resize)
    
    avg_conf_thresh = float(FLAGS.pixel_conf_threshold + FLAGS.link_conf_threshold)/2 

    global_step = slim.get_or_create_global_step()
    # with tf.name_scope('evaluation_%dx%d'%(FLAGS.eval_image_height, FLAGS.eval_image_width)):
    with tf.name_scope('evaluation_%dx%d'%(0000, 0000)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = False):
            image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
            image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                       out_shape = config.image_shape,
                                                       data_format = config.data_format,
                                                       do_resize = False,
                                                       is_training = False)
            b_image = tf.expand_dims(processed_image, axis = 0)

            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training = False)
            masks = pixel_link.tf_decode_score_map_to_mask_in_batch(
                net.pixel_pos_scores, net.link_pos_scores)
            
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore(
                tf.trainable_variables())
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()
        
    
    timer = [[], [], [], [], []]     ## load_image, pad_image, inference, cal_box, total
    saver = tf.train.Saver(var_list = variables_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)
        
        files = os.listdir(FLAGS.dataset_dir)
        
        for image_name in files:
            sp1 = time.time()
            file_path = os.path.join(FLAGS.dataset_dir, image_name)
            origin_image_data = cv2.imread(file_path)
            sp2 = time.time()          

            '''padding to avoid distort'''
            # image_data = cv_pad(image_data, config.image_shape)
            if FLAGS.scale_resize != 1:
                image_data = scale_resize(origin_image_data, FLAGS.scale_resize)
            else:
                image_data = origin_image_data
            sp3 = time.time()

            link_scores, pixel_scores, mask_vals = sess.run(
                    [net.link_pos_scores, net.pixel_pos_scores, masks],
                    feed_dict = {image: image_data})
            h, w, _ =image_data.shape
            sp4 = time.time()
            
            def resize(img):
                return cv2.resize(img, size=(w,h), interpolation = cv2.INTER_NEAREST)
            
            def get_bboxes(mask):
                return pixel_link.mask_to_bboxes(mask, image_data.shape)

            def points_to_contour(points):
                contours = [[list(p)]for p in points]
                return np.asarray(contours, dtype = np.int32)

            def points_to_contours(points):
                return np.asarray([points_to_contour(points)])
            
            def draw_bboxes(img, bboxes, color):
                for bbox in bboxes:
                    points = np.reshape(bbox, [4, 2])
                    cnts = points_to_contours(points)
                    cv2.drawContours(img, contours = cnts, 
                           idx = -1, color = color, border_width = 1)


            image_idx = 0
            pixel_score = pixel_scores[image_idx, ...]
            mask = mask_vals[image_idx, ...]

            bboxes_det = get_bboxes(mask)
            _bboxes_det = revert_dectbox(bboxes_det, FLAGS.scale_resize)
            sp5 = time.time()
            # print ">> bboxes_det:",type(bboxes_det), bboxes_det
            # print ">> _bboxes_det:",type(_bboxes_det), _bboxes_det            
            
            mask = resize(mask)
            pixel_score = resize(pixel_score)

            draw_bboxes(origin_image_data, _bboxes_det, (0, 0, 255))
            cv2.imwrite(os.path.join(FLAGS.output_dir, 'out_'+os.path.basename(file_path)), origin_image_data)

            nameID = image_name.split('.')[0]
            for bbox in _bboxes_det:
                # print "nameID, bbox", nameID, bbox
                _bbox = []
                for num in bbox:
                    _bbox.append(num)
                wfile.write("{}\t{}\t{}\n".format(nameID, avg_conf_thresh, _bbox))

            ## timer accumulate
            timer[0].append(sp2-sp1)
            timer[1].append(sp3-sp2)
            timer[2].append(sp4-sp3)
            timer[3].append(sp5-sp4)
            timer[4].append(sp5-sp1)
            print "{}:{}\t{}:{}\t{}:{}\t{}:{}\t{}:{}\n".format('Load', round(sp2-sp1,3), 'Pad', round(sp3-sp2,3), \
                        'Infer', round(sp4-sp3,3), 'Post', round(sp5-sp4,3), 'Total', round(sp5-sp1,3))
        print "\nAvg Timer Stat:"
        print "{}:{}\t{}:{}\t{}:{}\t{}:{}\t{}:{}\n".format('Load', round(np.mean(timer[0]),3), 'Pad', round(np.mean(timer[1]),3), \
            'Infer', round(np.mean(timer[2]),3), 'Post', round(np.mean(timer[3]),3), 'Total', round(np.mean(timer[4]),3))
        wfile.close()

        
def main(_):
    dataset = config_initialization()
    test()
    
    
if __name__ == '__main__':
    tf.app.run()
