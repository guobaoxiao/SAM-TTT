import os
import sys

import cv2
from tqdm import tqdm
import torch
import sod_metrics as M
import torch.nn.functional as F
from transformer_npz_2_gt import transformer2gt
from data_visual2 import datavisual
FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()


def _upsample_like(src, tar):
    src = torch.tensor(src, dtype=torch.float32)
    tar = torch.tensor(tar)
    src = F.interpolate(src.unsqueeze(0).unsqueeze(0), size=tar.shape, mode='bilinear')
    src = src.squeeze(0).squeeze(0).numpy()
    return src

def evaluate_for_datasets(mask_root, pred_root, dataset, task, data_file):
    mask_name_list = sorted(os.listdir(mask_root))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        mask_name_for_pred = mask_name.replace(".png", ".jpg")
        # mask_name_for_pred = mask_name
        pred_path = os.path.join(pred_root, mask_name_for_pred)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if len(pred.shape) != 2:
            pred = pred[:, :, 0]  # 返回(height, width)
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]
        pred = _upsample_like(pred, mask)
        assert pred.shape == mask.shape

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    print(
        'Task:', task, '; ',
        'Dataset:', dataset, '; ',
        'Smeasure:', sm.round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'maxFm:', fm['curve'].max().round(3),
        sep=''
    )

    with open(data_file, "a+") as f:
        print(
            'Task:', task, '; ',
            'Dataset:', dataset, '; ',
            'Smeasure:', sm.round(3), '; ',
            'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
            'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
            'wFmeasure:', wfm.round(3), '; ',
            'meanFm:', fm['curve'].mean().round(3), '; ',
            'MAE:', mae.round(3), '; ',
            file=f
        )

if __name__=='__main__':
    # datasets = ['CAMO', 'COD10K', 'NC4K']
    # task = '241022_train_generated_4conv_13_ch2_340'
    #
    # path = '../data/inference_npz/' + str(task)
    # save_path = '../data/inference_img/' + str(task)
    # transformer2gt(path, save_path)

    data_file = 'txt/241023_abla_visual.txt'
    # for i in range(len(datasets)):
    #     mask_root = '/data/Jenny/Media/COD/' + str(datasets[i]) + '/gt'
    #     pred_root = '../data/inference_img/' + str(task) + '/' + str(datasets[i]) + '/'
    #     evaluate_for_datasets(mask_root, pred_root, datasets[i], task, data_file)
    #
    output_file = 'img/250401_ablation.pdf'
    datavisual(data_file, output_file)