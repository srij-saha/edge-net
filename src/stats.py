import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

#------1. Illusion Magnitude computatation ------------#
def compute_illusion_magnitude_classic(prediction, dark_mask, light_mask):
    """ Computes the difference in the reconstructed region for one model on one classic illusion"""

    pred_dark = prediction[dark_mask == 1] * 255
    pred_light = prediction[light_mask == 1] * 255
    
    pred_dark_mean = pred_dark.mean()
    pred_light_mean = pred_light.mean()
    pred_diff = pred_light_mean - pred_dark_mean
    
    return {
        "pred_dark_mean": pred_dark_mean,
        "pred_light_mean": pred_light_mean,
        "pred_diff": pred_diff,
    }

def compute_illusion_magnitude_moon(pred_dark_img, pred_light_img, mask):
    """ Computes the absolute illusion magnitude for one model on one moon illusion."""
    # moon means
    pred_dark = pred_dark_img[mask == 1] * 255
    pred_light = pred_light_img[mask == 1] * 255
    pred_dark_mean = pred_dark.mean()
    pred_light_mean = pred_light.mean()
    
    pred_diff = pred_light_mean - pred_dark_mean
    
    return {
        "pred_dark_mean": pred_dark_mean,
        "pred_light_mean": pred_light_mean,
        "pred_diff": pred_diff,
    }

#------2. Within-illusion existence tests ------------#
def ttest_across_models(effects, popmean = 0.0):
    """ Run a one-sample t-test over the model-level illusion magnitudes."""
    effects = np.asarray(effects, dtype = float)
    n = len(effects)
    
    mean = effects.mean()
    
    if n > 1:
        std = effects.std(ddof = 1)
        sderr = std/np.sqrt(n)
    else:
        std = np.nan
        sderr = np.nan
    
    t_stat, p_value = ttest_1samp(effects, popmean = popmean)
    
    return {
        "mean": mean,
        "std": std,
        "sderr": sderr,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_models": n,
    }

def summarize_illusion_across_models(df, value_col="illusion_magnitude", group_cols = None):
    """
    Group a DataFrame by group_cols and run ttest_across_models on value_col within each group
    
    Returns a new DataFrame with one row per group and t-test summary stats.
    """
    
    if group_cols is None:
        group_cols = []
        
    rows =[]
    grouped = df.groupby(group_cols) if group_cols else [((), df)]
    
    for group_key, group_df in grouped:
        effects = group_df[value_col].to_numpy()
        stats = ttest_across_models(effects)
        
        row = {
            **{col: val for col, val in zip(group_cols, (group_key if isinstance(group_key, tuple) else (group_key,)))},
            **stats
        }
        rows.append(row)
    return pd.DataFrame(rows)

#------3. Parametric and ordinal trend tests ------------#
def _add_mapped_code(df, source_col, mapping, new_col):
    
    missing = set(df[source_col].unique()) - set(mapping.keys())
    if missing:
        raise ValueError(f"Missing mapping for {source_col}: {sorted(missing)}")
    
    df = df.copy()
    df[new_col] = df[source_col].map(mapping)
    return df

def spearman_ordinal_trend(df, ordinal_col, value_col):
    """
    Spearman rank correlation for an ordinal manipulation
    e.g. Purves Cornsweet ordering: inverted < horizontal < upright
    """
    sub = df[[ordinal_col, value_col]].copy()
        
    if sub.isna().any().any():
        raise ValueError(f"NaNs present in {ordinal_col} or {value_col}")
    
    x = pd.to_numeric(sub[ordinal_col], errors = "raise").to_numpy()
    y = pd.to_numeric(sub[value_col], errors = "raise").to_numpy()
    
    rho, p_value = spearmanr(x,y)
    
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n": int(len(sub)),
    }

def linear_parametric_trend(df, param_col, value_col, fe_col = None):
    """ 
    Linear trend test for a metric parametric manipulation
    
    IV = metric parametric manipulation
    e.g. white-illusion width levels
    
    model: value_col ~ param_col + (+ C(fe_col))
    """
    
    cols = [param_col, value_col] + ([fe_col] if fe_col else [])
    sub = df[cols].copy()
    
    if sub.isna().any().any():
        raise ValueError(f"NaNs present in required cols: {cols}")
    
    sub[param_col] = pd.to_numeric(sub[param_col], errors = "raise")
    sub[value_col] = pd.to_numeric(sub[value_col], errors = "raise")
    
    model_form = f"{value_col} ~ {param_col}"
    if fe_col:
        model_form += f" + C({fe_col})"
    
    model = smf.ols(model_form, data = sub).fit()
    
    return {
        "slope": float(model.params.get(param_col, np.nan)),
        "slope_p_value": float(model.pvalues.get(param_col, np.nan)),
        "r_squared": float(model.rsquared),
        "n": int(model.nobs),
        "fe_col": fe_col,
    }

#------4. Legacy and pixel-level analysis ------------#
def pixel_perm_test_single_image (image_pred, dark_mask, light_mask, n_permutations = 10000, seed = 42):
    pixels_dark_region = (image_pred[dark_mask == 1] * 255).astype(np.float64)
    pixels_light_region = (image_pred[light_mask == 1] * 255).astype(np.float64)
    
    num_pixels_dark_region = pixels_dark_region.size
    num_pixels_light_region = pixels_light_region.size
    
    pred_region_diff = np.mean(pixels_light_region) - np.mean(pixels_dark_region)
    
    rng = np.random.default_rng(seed)
    pooled_pixels = np.concatenate([pixels_dark_region, pixels_light_region])
    null_region_diffs = np.empty(n_permutations, dtype = np.float64)
    
    for i in range(n_permutations):
        perm_pixels = rng.permutation(pooled_pixels)
        perm_dark_region = perm_pixels[:num_pixels_dark_region]
        perm_light_region = perm_pixels[num_pixels_dark_region:]
        null_region_diffs[i] = np.mean(perm_light_region) - np.mean(perm_dark_region)
        
    p_value = (np.sum(np.abs(null_region_diffs) >= np.abs(pred_region_diff)) + 1)/ (n_permutations + 1)
    return pred_region_diff, p_value, null_region_diffs

def pixel_perm_test_anderson_winawer_moon(pred_dark, pred_light, mask, adj = False, n_permutations = 10000, seed = 42):
    pixels_dark_region = (pred_dark[mask == 1] * 255).astype(np.float64)
    pixels_light_region = (pred_light[mask == 1] * 255).astype(np.float64)
    
    num_pixels_dark_region = pixels_dark_region.size
    num_pixels_light_region = pixels_light_region.size
    
    if adj == True:
        mean_dark_bg = np.mean((pred_dark[mask != 1] * 255).astype(np.float64))
        mean_light_bg = np.mean((pred_light[mask != 1] * 255).astype(np.float64))
    
        pixels_dark_region = pixels_dark_region - mean_dark_bg
        pixels_light_region = pixels_light_region - mean_light_bg
    
    region_diff = np.mean(pixels_light_region) - np.mean(pixels_dark_region)
    
    rng = np.random.default_rng(seed)
    pooled_pixels = np.concatenate([pixels_dark_region, pixels_light_region])
    null_region_diffs = np.empty(n_permutations, dtype = np.float64)
    
    for i in range(n_permutations):
        perm_pixels = rng.permutation(pooled_pixels)
        perm_dark_region = perm_pixels[:num_pixels_dark_region]
        perm_light_region = perm_pixels[num_pixels_dark_region:]
        null_region_diffs[i] = np.mean(perm_light_region) - np.mean(perm_dark_region)
        
    p_value = (np.sum(np.abs(null_region_diffs) >= np.abs(region_diff)) + 1)/ (n_permutations + 1)
    return region_diff, p_value, null_region_diffs


def signflip_nonparametric_onesampletest(ds_ill_effects):
    obs_effects = np.asarray(ds_ill_effects, dtype = float)
    obs_mean = obs_effects.mean()
    n_models = obs_effects.size
    
    obs_sderr = np.std(obs_effects,ddof = 1)/np.sqrt(n_models)
    
    null_effects_means = np.fromiter(
        (np.mean(obs_effects * np.array(sign_flip)) for sign_flip in product([-1,1], repeat = n_models)),
        dtype = float,
        count = 2**n_models
    )
    
    #two-tailed p-value
    p_value = np.mean(np.abs(null_effects_means) >= abs(obs_mean))
    
    
    return {"mean_ill_magnitude": obs_mean, "sderr_ill_magnitude": obs_sderr, "p_value": p_value, "null_effects_means": null_effects_means}