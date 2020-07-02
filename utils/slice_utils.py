import os
import sys
import json
import numpy as np
from PIL import Image
import cv2

sys.path.append('../')
from data import BoneCellDetection
# from data.BoneCellDataset import *
from .augmentations import jaccard_numpy

def reverse_transform(boxes, labels, output_file):
    shapes = []
    target = { 'shapes': [] }
    for (box, label) in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        rect = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
        shapes.append({
            'label': idx_to_class[int(label)],
            'points': rect,
            'shape_type': 'polygon'
        })
    target = { 'shapes': shapes }
    with open(output_file, 'w') as f:
        # print(target)
        f.write(json.dumps(target))
        

def slice_anno(rect, targets, min_iou=0.1, max_iou=float('inf')):
    boxes = targets[:,:4]
    labels = targets[:,4]
    # overlap = jaccard_numpy(boxes, rect)

    overlap = jaccard_numpy(boxes, rect)

    # print('overlap={}'.format(overlap))
    # print('overlap min={}, overlap max={}'.format(overlap.min(), overlap.max()))
    # print('min_iou={}'.format(min_iou))
    # exit()

    # is min and max overlap constraint satisfied? if not try again
    if overlap.min() < min_iou and max_iou < overlap.max():
        return [], []

    # keep overlap with gt box IF center in sampled patch
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

    # mask in all gt boxes that above and to the left of centers
    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

    # mask in all gt boxes that under and to the right of centers
    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

    # mask in that both m1 and m2 are true
    mask = m1 * m2

    # have any valid boxes? try again if not
    if not mask.any():
        return [], []

    # take only matching gt boxes
    current_boxes = boxes[mask, :].copy()

    # take only matching gt labels
    current_labels = labels[mask]

    # should we use the box left and top corner or the crop's
    current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                      rect[:2])
    # adjust to crop (by substracting crop's left,top)
    current_boxes[:, :2] -= rect[:2]

    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                      rect[2:])
    # adjust to crop (by substracting crop's left,top)
    current_boxes[:, 2:] -= rect[:2]

    return current_boxes, current_labels

def slice(input_dir, inputfile, output_dir, k_w, k_h):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # assert np.sqrt(k)%1 == 0, # k must be some 
    dataset = BoneCellDetection(root=input_dir)
    img_idx = dataset.get_img_idx(inputfile)
    # for t in range(len(dataset)):
        # img_id = dataset.ids[t]
        # _im, target, orig_target, im_h, im_w = dataset.pull_item(t)
    # print(inputfile)
    _im, target, orig_target, im_h, im_w = dataset.pull_item(img_idx)
    np_targets = None
    if len(target) > 0:
        np_targets = np.array(target)
        # height, width, channels = im.shape
        np_targets[:, 0] *= im_w
        np_targets[:, 2] *= im_w
        np_targets[:, 1] *= im_h
        np_targets[:, 3] *= im_h
    # np_targets = np_targets
    
    # im_w, im_h = im.size
    # print(im_w, im_h)
    # im_w, im_h, _ = 400, 1000, 3
    box_w = int(round(im_w/k_w))
    box_h = int(round(im_h/k_h))
    print(box_w, im_w//k_w)
    print(box_h, im_h//k_h)
    # print(im_w, im_h)
    # print(box_w, box_h)
    # exit()
    fname, ext = inputfile.split('.')
    im = Image.open(os.path.join(input_dir, inputfile))
    for _i, i in enumerate(range(0, im_w-1, box_w)):
        print('i', _i, i)
        for _j, j in enumerate(range(0, im_h-1, box_h)):
            print('j', _j, j)
            # input("Press Enter to continue...")
            rect = (i, j, i+box_w, j+box_h)
            # print('rect')
            # print(rect)
            # print('all boxes')
            # print(np_targets[:,:4])
            curr_fname = fname + '_{}_{}.'.format(_i+1, _j+1)
            try:
                boxes, labels = slice_anno(rect, np_targets)
                reverse_transform(boxes, labels, os.path.join(output_dir, curr_fname + 'json'))
            except:
                pass
            
            a = im.crop(rect)
            a.save(os.path.join(output_dir, curr_fname + ext))

def stitch_image(img_arr, n_slices):
    """
    input is an nxn array of cv2 images.
    """
    img_2d_arr = []
    for j in range(n_slices):
        curr_col = []
        for i in range(n_slices):
            curr_col.append(img_arr[i*n_slices + j])
        img_2d_arr.append(curr_col)
    cols = [cv2.vconcat(img_2d_arr[0])]
    for i in range(1, len(img_2d_arr)):
        print('stitching row {}'.format(i))
        cols.append(cv2.vconcat(img_2d_arr[i]))
    return cv2.hconcat(cols)


def do_slice():
    # input_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test'
    # output_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test_sliced'
    # input_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test'
    # output_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test/MICHELLE_LABELED_TEST_SLICED'
    input_dir = '/Users/edock/Downloads/MICHELLE_LABELED_TEST'
    # output_dir = '/Users/edock/Downloads/SAPIR_LABELED_TEST/SAPIR_LABELED_TEST_SLICED'
    output_dir = '/Users/edock/Downloads/MICHELLE_LABELED_TEST/SAPIR_LABELED_TEST_SLICED'

    
    # image_list = [  'G4_01_01.png' ]
    with open(os.path.join(input_dir, 'file_list.txt'), 'r') as f:
        image_list = f.readlines()
    image_list = [f.rstrip()[:-4] + '.png' for f in image_list]
    print(image_list)
    # print(image_list)
    # exit()
    # image_list = [  'G4_01_01.png',
    #                 'G4_01_02.png',
    #                 'G4_01_03.png',
    #                 'G4_01_04.png',
    #                 'G4_02_01.png',
    #                 'G4_02_02.png',
    #                 'G4_02_03.png',
    #                 'G4_02_04.png',
    #                 'G4_03_01.png',
    #                 'G4_03_02.png',
    #                 'G4_03_03.png',
    #                 'G4_03_04.png',
    #                 'G4_04_02.png',
    #                 'G4_04_03.png']

    for img in image_list:
        slice(input_dir, img, output_dir, 2, 2)

if __name__ == '__main__':
    # dataset = BoneCellDetection(mode='SLICED_TEST_11')
    # for i in range(10):
    #     dataset.draw_item(i, drawRecs=True, drawPolygone=False)
    # dataset.draw_item(1, drawRecs=True, drawPolygone=False)
    # dataset.draw_item(2, drawRecs=True, drawPolygone=False)
    # dataset.draw_item(3, drawRecs=True, drawPolygone=False)
    # dataset.draw_item(4, drawRecs=True, drawPolygone=False)
    do_slice()