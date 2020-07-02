"""
Author: Edo Cohen-Karlik
"""
from __future__ import division
# import os.path as osp
import json
# import sys
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np


#from augmentations import SSDAugmentation, SSDBoneCellAugmentation

# ignore classes with label value -1.
BONE_CELL_CLASSES_MAP = {
    'p': 1,
    'g': 2,
    '0_1': 3,
    '0_2': 4,
}

# format: BGR
CLASS_COLOR_MAP = {
    'p': (200,0,180), # purple
    'g': (0,200,0), # green
    '0_1': (255,0,0), # blue
    '0_2': (0,0,255), # red
}

idx_to_class = {}
for k, v in BONE_CELL_CLASSES_MAP.items():
    if v != -1:
        idx_to_class[v] = k

def convert_circle_to_bbox(points, width, height):
    center = points[0]
    edge = points[1]
    radius = int(np.sqrt(np.power(center[0]-edge[0], 2) + np.power(center[1]-edge[1], 2)))
    # xmin, ymin, xmax, ymax
    xmin = (center[0] - radius) / width
    ymin = (center[1] - radius) / height
    xmax = (center[0] + radius) / width
    ymax = (center[1] + radius) / height
    return [xmin, ymin, xmax, ymax]

def convert_polygon_to_bbox(points, width, height):
    points = np.array(points)
    xmin = np.min(points[:,0]) / width
    ymin = np.min(points[:,1]) / height
    xmax = np.max(points[:,0]) / width
    ymax = np.max(points[:,1]) / height
    return [xmin, ymin, xmax, ymax]

def convert_shape_to_bbox(points, shape_type, width, height):
    if shape_type == 'circle':
        return convert_circle_to_bbox(points, width, height)
    elif shape_type == 'polygon':
        return convert_polygon_to_bbox(points, width, height)

def mark_area(mat, start_point, end_point):
    # print(start_point, end_point)
    mat[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 1
    return mat

def draw_bbox_on_img(img, bboxes, bbox_format='gt', get_area=False):
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    area_mat = np.zeros((img.shape[0], img.shape[1]))
    # print(img.shape)
    # print(area_mat.shape)
    # exit()
    img = img.astype(np.uint8).copy()
    height, width = img.shape[:2]
    if bbox_format == 'pred':
        width = height = 1.0
    for rec in bboxes:
        cls_idx = rec[-1]
        start_point = (int(rec[0]*width), int(rec[1]*height))
        end_point = (int(rec[2]*width), int(rec[3]*height))

        area_mat = mark_area(area_mat, start_point, end_point)
        img = cv2.rectangle(img, start_point, end_point, CLASS_COLOR_MAP[idx_to_class[cls_idx]], 2)

    # print(area_mat.sum()/(area_mat.shape[0]*area_mat.shape[1]))
    # area = area_mat.sum()/(area_mat.shape[0]*area_mat.shape[1])
    area = area_mat.mean()
    if get_area:
        return img, area
    return img


class BoneCellAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    # def __init__(self):
    #     self.class_to_ind = dict(zip(BONE_CELL_CLASSES, range(len(BONE_CELL_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        orig_res = []
        skipped = 0
        # for obj in target.iter('object'):
        for obj in target['shapes']:
            label_idx = BONE_CELL_CLASSES_MAP[obj['label']]
            if label_idx == -1:
                skipped += 1
                continue

            # pts = ['xmin', 'ymin', 'xmax', 'ymax']
            if obj['shape_type'] == 'polygon':
                orig_plgn = obj['points']
            else:
                orig_plgn = None
            bndbox = convert_shape_to_bbox(obj['points'], obj['shape_type'], width, height)
            bndbox.append(label_idx)
                
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            orig_res += [orig_plgn]
            ## img_id = target.find('filename').text[:-4]
        # print('skipped {} objects'.format(skipped))
        return res, orig_res # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class BoneCellDetection(data.Dataset):
    """BoneCell Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to BoneCell folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, transform=None, target_transform=BoneCellAnnotationTransform(),
                 dataset_name='BONECELL'):
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        # for line in open(osp.join(root, mode, mode + '.txt')):
        for f in os.listdir(root):
            if f.endswith('.png'):
                _line = f.split('.')
                fname = '.'.join(_line[:-1])
                # fname = os.path.join(*fname)
                self.ids.append(os.path.join(root, fname))
        print('loaded Bonecell dataset with {} images'.format(len(self.ids)))
        # for line in open(osp.join(root, 'file_list.txt')):
        #     _line = line.split('.')
        #     fname = '.'.join(_line[:-1])
        #     # fname = os.path.join(*fname)
        #     self.ids.append(osp.join(root, fname))

    def __getitem__(self, index):
        im, gt, orig_gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def get_img_idx(self, img_id):
        img_id, _ = img_id.split('.')
        for idx, _id in enumerate(self.ids):
            _id = _id.split('/')[-1]
            if img_id == _id:
                return idx

    def pull_item(self, index):
        img_id = self.ids[index]
        # print('image id: {}'.format(img_id))
        with open(img_id + '.json', 'r') as f:
            target = json.loads(f.read())
        img = cv2.imread(img_id + '.png')
        height, width, channels = img.shape
        if self.target_transform is not None:
            target, orig_target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1), target, orig_target, height, width

    def draw_gt(self, index, get_area=False):
        img, target, original_target, height, width = self.pull_item(index)
        return draw_bbox_on_img(img, target, get_area=get_area)

    def draw_pred(self, index, pred_bbox, get_area=False):
        img, target, original_target, height, width = self.pull_item(index)
        return draw_bbox_on_img(img, pred_bbox, bbox_format='pred', get_area=get_area)


class BoneCellInfer(BoneCellDetection):
    """BoneCell Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to BoneCell folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, root, transform, target_transform=None, dataset_name='INFER'):
        super().__init__(root=root, transform=transform,
                            target_transform=target_transform, dataset_name=dataset_name)
        
    def pull_item(self, index):
        img_id = self.ids[index]
        img = cv2.imread(img_id + '.png')
        height, width, channels = img.shape
        img, _, _ = self.transform(img, None, None)
        img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1), None, None, height, width
        draw_bbox_on_img

    def draw_gt(self, index, get_area=False):
        img, target, original_target, height, width = self.pull_item(index)
        return draw_bbox_on_img(img, [], get_area=False)

    def draw_pred(self, index, pred_bbox, get_area=False):
        img, target, original_target, height, width = self.pull_item(index)
        return draw_bbox_on_img(img, pred_bbox, bbox_format='pred', get_area=False)

# def base_transform(image, size, mean):
#     x = cv2.resize(image, (size, size)).astype(np.float32)
#     x -= mean
#     x = x.astype(np.float32)
#     return x


# class BaseTransform:
#     def __init__(self, size, mean):
#         self.size = size
#         self.mean = np.array(mean, dtype=np.float32)

#     def __call__(self, image, boxes=None, labels=None):
#         return base_transform(image, self.size, self.mean), boxes, labels