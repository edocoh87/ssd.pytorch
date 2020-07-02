import os
import sys
import numpy as np
import torch
from data import BaseTransform, BoneCellDetection, BoneCellInfer
# from data import BaseTransform, BoneCellDetection, BoneCellInfer, BONE_CELL_CLASSES_MAP
# from .slice_utils import *
from . import slice_utils
from layers import box_utils
from ssd import build_ssd

sys.path.append('../')
from eval import test_net
import pickle
import cv2

BONE_CELL_CLASSES_MAP = {
    'p': 1,
    '0_1': 3,
    '0_2': 4,
}

model = 'weights/bonecell_mean_0_0_0/ssd300_BONECELL_final.pth'
dataset_mean = (0, 0, 0)

def parse_predictions(results, idx, threshold):
    curr_img_recs = []
    for cls, cls_idx in BONE_CELL_CLASSES_MAP.items(): #class
        curr_cls_bboxes = results[cls_idx][idx]
        for curr_box in curr_cls_bboxes:
            if curr_box[-1] >= threshold:
                curr_img_recs.append(curr_box.tolist() + [cls_idx])
    return curr_img_recs

    
class ImageInfer:
    def __init__(self, img_dir, img_fname, slices_per_axis):
        self.img_dir = img_dir
        self.img_fname = img_fname
        self.sliced_img_save_dir = self.img_fname[:-4] + '_slices_{}'.format(slices_per_axis)
        self.output_dir = os.path.join(self.img_dir, self.sliced_img_save_dir)
        self.output_dir_name_template = self.img_fname[:-4] + '_{}_{}.png'
        self.slices_per_axis = slices_per_axis
        self.num_of_recs = []
        self.post_nms_num_of_recs = []
        # self.results_file_name = None
        self.results_file_name = 'detection_results.pkl'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            slice_utils.slice(self.img_dir, self.img_fname, self.output_dir, self.slices_per_axis, self.slices_per_axis)
        
        # self.dataset = BoneCellDetection(root=self.output_dir, transform=BaseTransform(300, dataset_mean))
        self.dataset = BoneCellInfer(root=self.output_dir, transform=BaseTransform(300, (0,0,0)))

    def stitch_image(self, img_arr):
        return slice_utils.stitch_image(img_arr, self.slices_per_axis)
    # def stitch_image(self, img_arr):
    #     img_2d_arr = []
    #     for j in range(self.slices_per_axis):
    #         curr_col = []
    #         for i in range(self.slices_per_axis):
    #             curr_col.append(img_arr[i*self.slices_per_axis + j])
    #         img_2d_arr.append(curr_col)
    #     cols = [cv2.vconcat(img_2d_arr[0])]
    #     for i in range(1, len(img_2d_arr)):
    #         print('stitching row {}'.format(i))
    #         cols.append(cv2.vconcat(img_2d_arr[i]))
    #     return cv2.hconcat(cols)
    
    def predict(self, net):
        test_net(
            save_folder=self.output_dir,
            results_file_name=self.results_file_name,
            net=net,
            cuda=False,
            dataset=self.dataset,
            transform=BaseTransform(net.size, dataset_mean),
            top_k=5,
            im_size=300,
            thresh=0.05)

    def parse_image(self, idx, curr_img_recs):
        bboxes = torch.Tensor(np.array([_box[:4] for _box in curr_img_recs]))
        scores = torch.Tensor(np.array([_box[4] for _box in curr_img_recs]))
        
        post_nms_img_recs = curr_img_recs
        if bboxes.shape[0] > 0:
            keep, count = box_utils.nms(bboxes, scores)
            keep = list(keep.numpy())
            if count < len(keep):
                post_nms_img_recs = [curr_img_recs[k] for k in keep]
        
        self.num_of_recs.append(len(curr_img_recs))
        self.post_nms_num_of_recs.append(len(post_nms_img_recs))
        gt_img = self.dataset.draw_gt(idx)
        pred_img = self.dataset.draw_pred(idx, pred_bbox=curr_img_recs)
        post_nms_pred_img = self.dataset.draw_pred(idx, pred_bbox=post_nms_img_recs)
        return gt_img, pred_img, post_nms_pred_img
        # cv2.imshow('image', gt_img)
        # cv2.waitKey(0)
        # cv2.imshow('image', pred_img)
        # cv2.waitKey(0)
        # cv2.imshow('image', post_nms_pred_img)
        # cv2.waitKey(0)
        # return (gt_img, pred_img, post_nms_pred_img)

    def parse_results(self, threshold=0.5):
        with open(os.path.join(self.output_dir, self.results_file_name), 'rb') as f:
            results = pickle.load(f)

        gt_img_arr = []
        pred_img_arr = []
        post_nms_pred_img_arr = []
        # [print(_id) for _id in self.dataset.ids]
        # exit()
        for i in range(self.slices_per_axis):
            for j in range(self.slices_per_axis):
                print(self.output_dir_name_template.format(i+1, j+1))
                idx = self.dataset.get_img_idx(self.output_dir_name_template.format(j+1, i+1))
                curr_img_recs = parse_predictions(results, idx, threshold)
                gt_img, pred_img, post_nms_pred_img = self.parse_image(idx, curr_img_recs)

                gt_img_arr.append(gt_img)
                pred_img_arr.append(pred_img)
                post_nms_pred_img_arr.append(post_nms_pred_img)
        return self.stitch_image(post_nms_pred_img_arr)
        # return self.stitch_image(gt_img_arr), self.stitch_image(pred_img_arr), self.stitch_image(post_nms_pred_img_arr)
            # _curr_img_recs = np.array(curr_img_recs).astype(np.int32)
            # print(_curr_img_recs)
            

if __name__=='__main__':
    # img_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test_sliced_small_subset'
    # img_fname = 'G4_03_02_2_2.png'
    # img_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/test_new_pred_per_img'
    img_base_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/zamzam_exp'
    # img_fname = '21_10-rl3.png'
    img_fname = 'RL_c4.png'
    imgInfer = ImageInfer(img_dir=img_dir, img_fname=img_fname, slices_per_axis=19)
    # dataset = BoneCellInfer(root=img_dir, transform=BaseTransform(300, (0,0,0)))
    num_classes = 5
    net = build_ssd('test', 300, num_classes) # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))
    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    imgInfer.predict(net)
    print('finished predicting.')
    pred_w_nms_img = imgInfer.parse_results()
    # gt_img, pred_img, pred_w_nms_img = imgInfer.parse_results()
    # pred_w_nms_img = imgInfer.parse_results()
    print(sum(imgInfer.num_of_recs))
    print(sum(imgInfer.post_nms_num_of_recs))
    resized_image = cv2.resize(pred_w_nms_img, (2000, 2000))
    cv2.imwrite(os.path.join(imgInfer.output_dir, 'pred_output.png'), resized_image)
    cv2.imshow('image', resized_image)
    cv2.waitKey(0)
    # cv2.imshow('image', pred_img)
    # cv2.waitKey(0)
    # cv2.imshow('image', pred_w_nms_img)
    # cv2.waitKey(0)
    