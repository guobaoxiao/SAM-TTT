import numpy as np
import scipy.ndimage
from utils.DSC_and_NSD import *
eval_test_model = 'test2'  # 'test2,test3,test4,test5,test6,test7,test8,test9,test10'
if eval_test_model == 'test1':
    # single pixels, 2mm away
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    mask_gt[50, 60, 70] = 1
    mask_pred[50, 60, 72] = 1
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))
elif eval_test_model == 'test2':
    # two cubes. cube 1 is 100x100x100 mm^3 and cube 2 is 102x100x100 mm^3
    mask_gt = np.zeros((100, 100, 100), np.uint8)
    mask_pred = np.zeros((100, 100, 100), np.uint8)
    spacing_mm = (2, 1, 1)
    mask_gt[0:50, :, :] = 1
    mask_pred[0:51, :, :] = 1
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm)
    print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))
    print("")
    print("expected average_distance_gt_to_pred = 1./6 * 2mm = {}mm".format(1. / 6 * 2))
    print("expected volumetric dice: {}".format(2. * 100 * 100 * 100 / (100 * 100 * 100 + 102 * 100 * 100)))
elif eval_test_model == 'test3':
    # test empty mask in prediction
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    mask_gt[50, 60, 70] = 1
    # mask_pred[50,60,72] = 1
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    print("average surface distance: {} mm".format(compute_average_surface_distance(surface_distances)))
    print("hausdorff (100%):         {} mm".format(compute_robust_hausdorff(surface_distances, 100)))
    print("hausdorff (95%):          {} mm".format(compute_robust_hausdorff(surface_distances, 95)))
    print("surface overlap at 1mm:   {}".format(compute_surface_overlap_at_tolerance(surface_distances, 1)))
    print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))
elif eval_test_model == 'test4':
    # test empty mask in ground truth
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    # mask_gt[50,60,70] = 1
    mask_pred[50, 60, 72] = 1
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    print("average surface distance: {} mm".format(compute_average_surface_distance(surface_distances)))
    print("hausdorff (100%):         {} mm".format(compute_robust_hausdorff(surface_distances, 100)))
    print("hausdorff (95%):          {} mm".format(compute_robust_hausdorff(surface_distances, 95)))
    print("surface overlap at 1mm:   {}".format(compute_surface_overlap_at_tolerance(surface_distances, 1)))
    print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))
elif eval_test_model == 'test5':
    # test both masks empty
    mask_gt = np.zeros((128, 128, 128), np.uint8)
    mask_pred = np.zeros((128, 128, 128), np.uint8)
    # mask_gt[50,60,70] = 1
    # mask_pred[50,60,72] = 1
    surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    print("average surface distance: {} mm".format(compute_average_surface_distance(surface_distances)))
    print("hausdorff (100%):         {} mm".format(compute_robust_hausdorff(surface_distances, 100)))
    print("hausdorff (95%):          {} mm".format(compute_robust_hausdorff(surface_distances, 95)))
    print("surface overlap at 1mm:   {}".format(compute_surface_overlap_at_tolerance(surface_distances, 1)))
    print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))