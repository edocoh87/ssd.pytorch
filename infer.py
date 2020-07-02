import os
import numpy as np
import pandas as pd
import torch
from data import BaseTransform, BoneCellDetection, BoneCellInfer
# from data import BaseTransform, BoneCellDetection, BoneCellInfer, BONE_CELL_CLASSES_MAP
from utils import slice_utils
from layers import box_utils
from ssd import build_ssd
import pickle
import cv2
from utils.eval_utils import ImageInfer


model = 'weights/bonecell_mean_0_0_0/ssd300_BONECELL_final.pth'
dataset_mean = (0, 0, 0)

file_list = ['EXP1/4.6,EPO,c2/4_01',
            'EXP1/4.6,EPO,c2/4_02',
            'EXP1/4.6,EPO,c2/4_03',
            'EXP1/4.6,EPO,c2/4_04',
            'EXP1/4.6,EPO,c2/4_05',
            'EXP1/4.6,EPO,c2/4_06',
            'EXP1/4.6,EPO,c2/4_07',
            'EXP1/4.6,EPO,c2/4_08',
            'EXP1/4.6,EPO,c2/4_09',
            'EXP1/4.6,EPO,c3/5_01',
            'EXP1/4.6,EPO,c3/5_02',
            'EXP1/4.6,EPO,c3/5_03',
            'EXP1/4.6,EPO,c3/5_04',
            'EXP1/4.6,EPO,c3/5_05',
            'EXP1/4.6,EPO,c3/5_06',
            'EXP1/4.6,EPO,c3/5_07',
            'EXP1/4.6,EPO,c3/5_08',
            'EXP1/4.6,EPO,c3/5_09',
            'EXP1/4.6,EPO,c4/6_01',
            'EXP1/4.6,EPO,c4/6_02',
            'EXP1/4.6,EPO,c4/6_03',
            'EXP1/4.6,EPO,c4/6_04',
            'EXP1/4.6,EPO,c4/6_05',
            'EXP1/4.6,EPO,c4/6_06',
            'EXP1/4.6,EPO,c4/6_07',
            'EXP1/4.6,EPO,c4/6_08',
            'EXP1/4.6,EPO,c4/6_09',
            'EXP1/4.6-ARA,g2/7_01',
            'EXP1/4.6-ARA,g2/7_02',
            'EXP1/4.6-ARA,g2/7_03',
            'EXP1/4.6-ARA,g2/7_04',
            'EXP1/4.6-ARA,g2/7_05',
            'EXP1/4.6-ARA,g2/7_06',
            'EXP1/4.6-ARA,g2/7_07',
            'EXP1/4.6-ARA,g2/7_08',
            'EXP1/4.6-ARA,g2/7_09',
            'EXP1/4.6-ARA,g3/8_01',
            'EXP1/4.6-ARA,g3/8_02',
            'EXP1/4.6-ARA,g3/8_03',
            'EXP1/4.6-ARA,g3/8_04',
            'EXP1/4.6-ARA,g3/8_05',
            'EXP1/4.6-ARA,g3/8_06',
            'EXP1/4.6-ARA,g3/8_07',
            'EXP1/4.6-ARA,g3/8_08',
            'EXP1/4.6-ARA,g3/8_09',
            'EXP1/4.6-ARA,g4/9_01',
            'EXP1/4.6-ARA,g4/9_02',
            'EXP1/4.6-ARA,g4/9_03',
            'EXP1/4.6-ARA,g4/9_04',
            'EXP1/4.6-ARA,g4/9_05',
            'EXP1/4.6-ARA,g4/9_06',
            'EXP1/4.6-ARA,g4/9_07',
            'EXP1/4.6-ARA,g4/9_08',
            'EXP1/4.6-ARA,g4/9_09',
            'EXP1/4.6-RL,b2/8_01',
            'EXP1/4.6-RL,b2/8_02',
            'EXP1/4.6-RL,b2/8_03',
            'EXP1/4.6-RL,b2/8_04',
            'EXP1/4.6-RL,b2/8_05',
            'EXP1/4.6-RL,b2/8_06',
            'EXP1/4.6-RL,b2/8_07',
            'EXP1/4.6-RL,b2/8_08',
            'EXP1/4.6-RL,b2/8_09',
            'EXP1/4.6-RL,b3/8_01',
            'EXP1/4.6-RL,b3/8_02',
            'EXP1/4.6-RL,b3/8_03',
            'EXP1/4.6-RL,b3/8_04',
            'EXP1/4.6-RL,b3/8_05',
            'EXP1/4.6-RL,b3/8_06',
            'EXP1/4.6-RL,b3/8_07',
            'EXP1/4.6-RL,b3/8_08',
            'EXP1/4.6-RL,b3/8_09',
            'EXP1/4.6-RL,b4/9_01',
            'EXP1/4.6-RL,b4/9_02',
            'EXP1/4.6-RL,b4/9_03',
            'EXP1/4.6-RL,b4/9_04',
            'EXP1/4.6-RL,b4/9_05',
            'EXP1/4.6-RL,b4/9_06',
            'EXP1/4.6-RL,b4/9_07',
            'EXP1/4.6-RL,b4/9_08',
            'EXP1/4.6-RL,b4/9_09',
            'EXP2/ara,f4/13_01',
            'EXP2/ara,f4/13_02',
            'EXP2/ara,f4/13_03',
            'EXP2/ara,f4/13_04',
            'EXP2/ara,f4/13_05',
            'EXP2/ara,f4/13_06',
            'EXP2/ara,f4/13_07',
            'EXP2/ara,f4/13_08',
            'EXP2/ara,f4/13_09',
            'EXP2/ara,g2/14_01',
            'EXP2/ara,g2/14_02',
            'EXP2/ara,g2/14_03',
            'EXP2/ara,g2/14_04',
            'EXP2/ara,g2/14_05',
            'EXP2/ara,g2/14_06',
            'EXP2/ara,g2/14_07',
            'EXP2/ara,g2/14_08',
            'EXP2/ara,g2/14_09',
            'EXP2/ara,g4/15_01',
            'EXP2/ara,g4/15_02',
            'EXP2/ara,g4/15_03',
            'EXP2/ara,g4/15_04',
            'EXP2/ara,g4/15_05',
            'EXP2/ara,g4/15_06',
            'EXP2/ara,g4/15_07',
            'EXP2/ara,g4/15_08',
            'EXP2/ara,g4/15_09',
            'EXP2/epo,d2/16_01',
            'EXP2/epo,d2/16_02',
            'EXP2/epo,d2/16_03',
            'EXP2/epo,d2/16_04',
            'EXP2/epo,d2/16_05',
            'EXP2/epo,d2/16_06',
            'EXP2/epo,d2/16_07',
            'EXP2/epo,d2/16_08',
            'EXP2/epo,d2/16_09',
            'EXP2/epo,d3/17_01',
            'EXP2/epo,d3/17_02',
            'EXP2/epo,d3/17_03',
            'EXP2/epo,d3/17_04',
            'EXP2/epo,d3/17_05',
            'EXP2/epo,d3/17_06',
            'EXP2/epo,d3/17_07',
            'EXP2/epo,d3/17_08',
            'EXP2/epo,d3/17_09',
            'EXP2/epo,d4/18_01',
            'EXP2/epo,d4/18_02',
            'EXP2/epo,d4/18_03',
            'EXP2/epo,d4/18_04',
            'EXP2/epo,d4/18_05',
            'EXP2/epo,d4/18_06',
            'EXP2/epo,d4/18_07',
            'EXP2/epo,d4/18_08',
            'EXP2/epo,d4/18_09',
            'EXP2/rl,c3/19_01',
            'EXP2/rl,c3/19_02',
            'EXP2/rl,c3/19_03',
            'EXP2/rl,c3/19_04',
            'EXP2/rl,c3/19_05',
            'EXP2/rl,c3/19_06',
            'EXP2/rl,c3/19_07',
            'EXP2/rl,c3/19_08',
            'EXP2/rl,c3/19_09',
            'EXP2/rl,c4/19_01',
            'EXP2/rl,c4/19_02',
            'EXP2/rl,c4/19_03',
            'EXP2/rl,c4/19_04',
            'EXP2/rl,c4/19_05',
            'EXP2/rl,c4/19_06',
            'EXP2/rl,c4/19_07',
            'EXP2/rl,c4/19_08',
            'EXP2/rl,c4/19_09',
            'EXP3/21.10-ara1/21_01',
            'EXP3/21.10-ara1/21_02',
            'EXP3/21.10-ara1/21_03',
            'EXP3/21.10-ara1/21_04',
            'EXP3/21.10-ara1/21_05',
            'EXP3/21.10-ara1/21_06',
            'EXP3/21.10-ara1/21_07',
            'EXP3/21.10-ara1/21_08',
            'EXP3/21.10-ara1/21_09',
            'EXP3/21.10-ara2/22_01',
            'EXP3/21.10-ara2/22_02',
            'EXP3/21.10-ara2/22_03',
            'EXP3/21.10-ara2/22_04',
            'EXP3/21.10-ara2/22_05',
            'EXP3/21.10-ara2/22_06',
            'EXP3/21.10-ara2/22_07',
            'EXP3/21.10-ara2/22_08',
            'EXP3/21.10-ara2/22_09',
            'EXP3/21.10-ara3/23_01',
            'EXP3/21.10-ara3/23_02',
            'EXP3/21.10-ara3/23_03',
            'EXP3/21.10-ara3/23_04',
            'EXP3/21.10-ara3/23_05',
            'EXP3/21.10-ara3/23_06',
            'EXP3/21.10-ara3/23_07',
            'EXP3/21.10-ara3/23_08',
            'EXP3/21.10-ara3/23_09',
            'EXP3/21.10-epo1/24_01',
            'EXP3/21.10-epo1/24_02',
            'EXP3/21.10-epo1/24_03',
            'EXP3/21.10-epo1/24_04',
            'EXP3/21.10-epo1/24_05',
            'EXP3/21.10-epo1/24_06',
            'EXP3/21.10-epo1/24_07',
            'EXP3/21.10-epo1/24_08',
            'EXP3/21.10-epo1/24_09',
            'EXP3/21.10-epo2/25_01',
            'EXP3/21.10-epo2/25_02',
            'EXP3/21.10-epo2/25_03',
            'EXP3/21.10-epo2/25_04',
            'EXP3/21.10-epo2/25_05',
            'EXP3/21.10-epo2/25_06',
            'EXP3/21.10-epo2/25_07',
            'EXP3/21.10-epo2/25_08',
            'EXP3/21.10-epo2/25_09',
            'EXP3/21.10-epo3/26_01',
            'EXP3/21.10-epo3/26_02',
            'EXP3/21.10-epo3/26_03',
            'EXP3/21.10-epo3/26_04',
            'EXP3/21.10-epo3/26_05',
            'EXP3/21.10-epo3/26_06',
            'EXP3/21.10-epo3/26_07',
            'EXP3/21.10-epo3/26_08',
            'EXP3/21.10-epo3/26_09',
            'EXP3/21.10-rl1/27_01',
            'EXP3/21.10-rl1/27_02',
            'EXP3/21.10-rl1/27_03',
            'EXP3/21.10-rl1/27_04',
            'EXP3/21.10-rl1/27_05',
            'EXP3/21.10-rl1/27_06',
            'EXP3/21.10-rl1/27_07',
            'EXP3/21.10-rl1/27_08',
            'EXP3/21.10-rl1/27_09',
            'EXP3/21.10-rl2/28_01',
            'EXP3/21.10-rl2/28_02',
            'EXP3/21.10-rl2/28_03',
            'EXP3/21.10-rl2/28_04',
            'EXP3/21.10-rl2/28_05',
            'EXP3/21.10-rl2/28_06',
            'EXP3/21.10-rl2/28_07',
            'EXP3/21.10-rl2/28_08',
            'EXP3/21.10-rl2/28_09',
            'EXP3/21.10-rl3/29_01',
            'EXP3/21.10-rl3/29_02',
            'EXP3/21.10-rl3/29_03',
            'EXP3/21.10-rl3/29_04',
            'EXP3/21.10-rl3/29_05',
            'EXP3/21.10-rl3/29_06',
            'EXP3/21.10-rl3/29_07',
            'EXP3/21.10-rl3/29_08',
            'EXP3/21.10-rl3/29_09']

# file_list = ['EXP3/21.10-rl3/29_09']
# file_list = ['EXP1/4.6,EPO,c2/4_01']
file_list = ['G4_01_01',
             'G4_01_02',
             'G4_01_03',
             'G4_01_04',
             'G4_02_01',
             'G4_02_02',
             'G4_02_03',
             'G4_02_04',
             'G4_03_01',
             'G4_03_02',
             'G4_03_03',
             'G4_03_04',
             'G4_04_02',
             'G4_04_03']
# file_list = ['G4_02_03']
# file_list = ['G4_01_03_rotate_1', 'G4_01_03_rotate_2', 'G4_01_03_rotate_3', 'G4_01_03_rotate_4']
count_per_image = {}
# img_base_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/zamzam_exp/'

img_base_dir = '/Users/edock/Work/bone_cell/data/BoneCellData/TEST_IMAGES/ZAMZAM_LABELED_TEST/'
slices_per_axis = 3
output_dir = os.path.join('/Users/edock/Work/ssd.pytorch/inference_results/ZAMZAM_TEST', 'output_slice_{}'.format(slices_per_axis))
threshold = 0.5
num_classes = 5
net = build_ssd('test', 300, num_classes) # initialize SSD
# net.load_state_dict(torch.load(args.trained_model))
net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
net.eval()
print('Finished loading model!')

cells_per_file = []
for file in file_list:
    if '/' in file:
        _file = '/'.join(file.split('/')[:-1])
        fname = file.split('/')[-1] + '.png'
    else:
        _file = ''
        fname = file + '.png'
    img_dir = img_base_dir + _file
    imgInfer = ImageInfer(img_dir=img_dir, img_fname=fname, slices_per_axis=slices_per_axis)

    print('finished loading {}'.format(file))
    imgInfer.predict(net)
    print('finished predicting.')
    pred_w_nms_img = imgInfer.parse_results(threshold=threshold)
    # gt_img, pred_img, pred_w_nms_img = imgInfer.parse_results()
    # pred_w_nms_img = imgInfer.parse_results()
    # print(sum(imgInfer.num_of_recs))
    # print(sum(imgInfer.post_nms_num_of_recs))
    cells_per_file.append(sum(imgInfer.post_nms_num_of_recs))
    resized_image = cv2.resize(pred_w_nms_img, (2000, 2000))
    if not os.path.exists(os.path.join(output_dir, _file)):
        os.makedirs(os.path.join(output_dir, _file))
    cv2.imwrite(os.path.join(output_dir, _file, 'pred_output_' + fname + '_slice_{}.png'.format(slices_per_axis)), resized_image)
    # resized_image = cv2.resize(pred_w_nms_img, (1000, 1000))
    # cv2.imshow('image', resized_image)
    # cv2.waitKey(0)
    # im = Image.open(file + '.gif')
    # im.save(file + ".png")
    # im.show()
df = pd.DataFrame({'files': file_list, 'counts': cells_per_file})
df.to_csv(os.path.join(output_dir, 'results_thrs_{}.csv'.format(slices_per_axis, threshold)))