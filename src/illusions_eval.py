import numpy as np
from . import io_utils
from .models_registry import parse_model_id
from .model_inference import run_test
from .stats import compute_illusion_magnitude_classic, compute_illusion_magnitude_moon

def eval_classic_single_edge(model_id, model, transform, root_folder, illusion_name, multiplier = 1.0, inputMin = -1, inputMax = 1, rotate90 = None, flip = None, plot = False):
    """ 
    Run one edge-based model on one classic illusion and return per-model metrics
    """
    meta = parse_model_id(model_id)
    
    illusion_file, dark_mask, light_mask = io_utils.get_files(root_folder, illusion_name, rotate90, flip)
    
    ground_truth, input_image, prediction, mae = run_test (model, transform, illusion_file,
                                                           multiplier = multiplier,
                                                           rotate90 = rotate90,
                                                           inputMin = inputMin,
                                                           inputMax = inputMax,
                                                           flip = flip,
                                                           plot = plot,
                                                          )
    
    gt_stats = compute_illusion_magnitude_classic(ground_truth, dark_mask, light_mask) 
    pred_stats = compute_illusion_magnitude_classic(prediction, dark_mask, light_mask)
    
    return {
        **meta,
        "illusion_name": illusion_name,
        "mae": float(mae),
        "gt_dark_mean": gt_stats["pred_dark_mean"],
        "gt_light_mean": gt_stats["pred_light_mean"],
        "gt_diff" : gt_stats["pred_diff"],
        **pred_stats,
    }

def eval_moon_single_edge(model_id, model, transform, root_folder, illusion_name, multiplier = 1.0, inputMin = -1, inputMax = 1, rotate90 = None, flip = None, plot = False):
    """
    Run one edge-based model on a Anderson-Winawer moon illusion and return per-model metrics
    """
    meta = parse_model_id(model_id)
    
    dark_illusion_file, light_illusion_file, mask = io_utils.get_anderson_winawer_moon_files(root_folder, illusion_name, rotate90 = rotate90, flip = flip)
    
    gt_dark, input_dark, pred_dark, mae_dark = run_test(
        model,
        transform,
        dark_illusion_file,
        multiplier = multiplier,
        inputMin = inputMin,
        inputMax = inputMax,
        rotate90 = rotate90,
        flip = flip,
        plot = plot,
    )
    
    gt_light, input_light, pred_light, mae_light = run_test(
        model,
        transform,
        light_illusion_file,
        multiplier = multiplier,
        inputMin = inputMin,
        inputMax = inputMax,
        rotate90 = rotate90,
        flip = flip,
        plot = plot,
    )
    
    raw_gt_stats = compute_illusion_magnitude_moon(gt_dark, gt_light, mask)
    gt_stats = {"gt_dark_mean": raw_gt_stats["pred_dark_mean"],
                "gt_light_mean": raw_gt_stats["pred_light_mean"],
                "gt_diff": raw_gt_stats["pred_diff"],
               }
    
    pred_stats = compute_illusion_magnitude_moon(
        pred_dark, pred_light, mask
    )
    
    return {
        **meta,
        "illusion_name": illusion_name,
        "mae_dark": float(mae_dark),
        "mae_light": float(mae_light),
        "mae_sum": float(mae_dark + mae_light),
        **gt_stats,
        **pred_stats,
    }

def eval_classic_single_denoise(model_id, model, gauss_transforms, root_folder, illusion_name,inputMin = -1, inputMax = 1, rotate90 = None, flip = None, plot = False):
    """
    Run one denoising model on one classic illusion using multiple Gaussian-noise transforms, average the reconstructions, and return per-model metrics.
    """
    
    meta = parse_model_id(model_id)
    
    illusion_file, dark_mask, light_mask = io_utils.get_files(root_folder, illusion_name, rotate90, flip)
    
    num_transforms = len(gauss_transforms)
    preds = []
    maes = []
    gt_ref = None
    
    for idx, transform in enumerate(gauss_transforms):
        ground_truth, input_image, prediction, mae = run_test(
            model = model,
            transform = transform,
            image_file = illusion_file,
            inputMin = inputMin, 
            inputMax = inputMax,
            rotate90 = rotate90,
            flip = flip,
            plot = plot,
        )
        
        if idx == 0:
            gt_ref = ground_truth
        preds.append(prediction)
        maes.append(mae)
    
    avg_prediction = np.mean(np.stack(preds, axis = 0), axis = 0)
    avg_mae = float(np.mean(maes))
    
    raw_gt_stats = compute_illusion_magnitude_classic(
        gt_ref, dark_mask, light_mask
    )
    
    gt_stats = {
        "gt_dark_mean": raw_gt_stats["pred_dark_mean"],
        "gt_light_mean": raw_gt_stats["pred_light_mean"],
        "gt_diff": raw_gt_stats["pred_diff"],
    }
    
    pred_stats = compute_illusion_magnitude_classic(
        avg_prediction, dark_mask, light_mask
    )
    
    return {
        **meta,
        "illusion_name": illusion_name,
        "avg_mae": avg_mae,
        **gt_stats,
        **pred_stats,
    }

def eval_moon_single_denoise(model_id, model, gauss_transforms, root_folder, illusion_name,  inputMin = -1, inputMax = 1, rotate90=None, flip = None, plot = False):
    """
    Run one denoising model on one instance of the Anderson-Winawer Illusion using multiple Gaussian transforms. Average dark and light
    reconstructions separately and return absolute and adjusted illusion magnitudes for the model
    """
    
    meta = parse_model_id(model_id)
    
    dark_illusion_file, light_illusion_file, mask = io_utils.get_anderson_winawer_moon_files(
        root_folder, illusion_name, rotate90=rotate90, flip=flip
    )
    
    preds_dark = []
    preds_light = []
    maes_dark = []
    maes_light = []
    gt_dark_ref = None
    gt_light_ref = None
    
    for idx, transform in enumerate(gauss_transforms):
        gt_dark, input_dark, pred_dark, mae_dark = run_test(
            model = model,
            transform = transform,
            image_file = dark_illusion_file,
            inputMin = inputMin,
            inputMax = inputMax,
            rotate90=rotate90,
            flip = flip,
            plot = plot
        )
        
        gt_light, input_light, pred_light, mae_light = run_test(
            model = model,
            transform = transform,
            image_file = light_illusion_file,
            inputMin = inputMin,
            inputMax = inputMax,
            rotate90 = rotate90,
            flip = flip,
            plot = plot
        )
        
        if idx == 0:
            gt_dark_ref = gt_dark
            gt_light_ref = gt_light
            
        preds_dark.append(pred_dark)
        preds_light.append(pred_light)
        maes_dark.append(mae_dark)
        maes_light.append(mae_light)
        
    avg_pred_dark = np.mean(np.stack(preds_dark, axis = 0), axis = 0)
    avg_pred_light = np.mean(np.stack(preds_light, axis = 0), axis = 0)
    avg_mae_dark = np.mean(maes_dark)
    avg_mae_light = np.mean(maes_light)
    avg_mae_sum = avg_mae_dark + avg_mae_light
    
    raw_gt_stats = compute_illusion_magnitude_moon(
        gt_dark_ref, gt_light_ref, mask
    )
    
    gt_stats = {
        "gt_dark_mean": raw_gt_stats["pred_dark_mean"],
        "gt_light_mean": raw_gt_stats["pred_light_mean"],
        "gt_diff": raw_gt_stats["pred_diff"],
    }
    
    pred_stats = compute_illusion_magnitude_moon(
        avg_pred_dark, avg_pred_light, mask
    )
    
    return {
        **meta,
        "illusion_name": illusion_name,
        "avg_mae_dark": avg_mae_dark,
        "avg_mae_light": avg_mae_light,
        "avg_mae_sum": avg_mae_sum,
        **gt_stats,
        **pred_stats,
    }

def eval_illusion_many_models(models_dict, eval_fn, **shared_kwags):
    """
    Evaluate a set of models on a single illusion using a given function
    models_dict: dict mapping model_id -> loaded model
    eval_fn: eval_classic_single_edge, eval_moon_single_edge, eval_classic_single_denoise or eval_moon_single_denoise
    shared_kwags: args shared across all models (root_folder, illusion_name, etc.)
    
    returns: list of result dict/model
    """
    
    results = []
    for model_id, model in models_dict.items():
        res = eval_fn(
            model_id= model_id,
            model=model,
            **shared_kwags,
        )
        results.append(res)
    return results
    
    
    
                             

                