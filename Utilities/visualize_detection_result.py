#encoding utf-8

import numpy as np
from postprocess import images


def draw_bbox(image_data, line, color):
    line = line.replace('\xef\xbb\xbf', '')
    data = line.split(',');
    points = [int(v) for v in data[0:8]]
    points = np.reshape(points, (4, 2))
    cnts = images.points_to_contours(points)
    images.draw_contours(image_data, cnts, -1, color = color, border_width = 3)
    
       
def visualize(image_root, det_root, output_root, gt_root = None):
    def read_gt_file(image_name):
        gt_file = os.path.join(gt_root, 'gt_%s.txt'%(image_name))
        return open(gt_file, 'rU').readlines()

    def read_det_file(image_name):
        det_file = os.path.join(det_root, 'res_%s.txt'%(image_name))
        return open(det_file, 'rU').readlines()
    
    def read_image_file(image_name):
        return images.imread(os.path.join(image_root, image_name))
    
    image_names = [] 
    _image_names = os.listdir(image_root)
    for name in _image_names:
        if name.endswith('jpg'):
            image_name.append(name)
    for image_idx, image_name in enumerate(image_names):
        print '%d / %d: %s'%(image_idx + 1, len(image_names), image_name)
        image_data = read_image_file(image_name) # in BGR
        image_name = image_name.split('.')[0]
        det_image = image_data.copy()
        det_lines = read_det_file(image_name)
        for line in det_lines:
            draw_bbox(det_image, line, color = images.COLOR_GREEN)
        output_path = os.path.join(output_root, '%s_pred.jpg'%(image_name))
        images.imwrite(output_path, det_image)
        print "Detection result has been written to ", os.path.abspath(output_path)
        
        if gt_root is not None:
            gt_lines = read_gt_file(image_name)
            for line in gt_lines:
                draw_bbox(image_data, line, color = images.COLOR_GREEN)
            images.imwrite(os.path.join(output_root, '%s_gt.jpg'%(image_name)), image_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize detection result of pixel_link')
    parser.add_argument('--image', type=str, required = True,help='the directory of test image')
    parser.add_argument('--gt', type=str, default=None,help='the directory of ground truth txt files')
    parser.add_argument('--det', type=str, required = True, help='the directory of detection result')
    parser.add_argument('--output', type=str, required = True, help='the directory to store images with bboxes')
    
    args = parser.parse_args()
    print('**************Arguments*****************')
    print(args)
    print('****************************************')
    visualize(image_root = args.image, gt_root = args.gt, det_root = args.det, output_root = args.output)
