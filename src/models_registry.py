import torch
import models

pretrained_models_edgenets = {
          '1k_ds1_edge_r1': {'weights': models.MUNet_v10_Weights.GAA_ORIG, 'model' : None},
          '1k_ds1_edge_r2': {'weights': models.MUNet_v10_Weights.GAA_REP1, 'model' : None},
          '1k_ds1_edge_r3': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP1, 'model' : None},
          '1k_ds1_edge_r4': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP2, 'model' : None},
          '1k_ds1_edge_r5': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP3, 'model' : None},
          '1k_ds1_edge_r6': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP4, 'model' : None},
          '1k_ds1_edge_r7': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP5, 'model' : None},
          '1k_ds1_edge_r8': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS1_REP6, 'model' : None},
          '1k_ds2_edge_r1': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP1, 'model': None},
          '1k_ds2_edge_r2': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP2, 'model': None},
          '1k_ds2_edge_r3': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP3, 'model': None},
          '1k_ds2_edge_r4': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP4, 'model': None},
          '1k_ds2_edge_r5': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP5, 'model': None},
          '1k_ds2_edge_r6': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP6, 'model': None},
          '1k_ds2_edge_r7': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP7, 'model': None},
          '1k_ds2_edge_r8': {'weights': models.MUNet_v10_Weights.SS_EDGE_DS2_REP8, 'model': None},
          '8k_ds1_edge_r1': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP1, 'model' : None},
          '8k_ds1_edge_r2': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP2, 'model' : None},
          '8k_ds1_edge_r3': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP3, 'model' : None},
          '8k_ds1_edge_r4': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP4, 'model' : None},
          '8k_ds1_edge_r5': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP5, 'model' : None},
          '8k_ds1_edge_r6': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP6, 'model' : None},
          '8k_ds1_edge_r7': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP7, 'model' : None},
          '8k_ds1_edge_r8': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_b_REP8, 'model' : None},
          '8k_ds2_edge_r1': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP1, 'model' : None},
          '8k_ds2_edge_r2': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP2, 'model' : None},
          '8k_ds2_edge_r3': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP3, 'model' : None},
          '8k_ds2_edge_r4': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP4, 'model' : None},
          '8k_ds2_edge_r5': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP5, 'model' : None},
          '8k_ds2_edge_r6': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP6, 'model' : None},
          '8k_ds2_edge_r7': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP7, 'model' : None},
          '8k_ds2_edge_r8': {'weights': models.MUNet_v10_Weights.SS_EDGE_8k_c_REP8, 'model' : None},
         }

pretrained_models_denoisenets = { 
                                 '1k_ds1_denoise_10_r1': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP1, 'model' : None},
                                 '1k_ds1_denoise_10_r2': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP2, 'model' : None},
                                 '1k_ds1_denoise_10_r3': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP3, 'model' : None},
                                 '1k_ds1_denoise_10_r4': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP4, 'model' : None},
                                 '1k_ds1_denoise_10_r5': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP5, 'model' : None},
                                 '1k_ds1_denoise_10_r6': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP6, 'model' : None},
                                 '1k_ds1_denoise_10_r7': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP7, 'model' : None},
                                 '1k_ds1_denoise_10_r8': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_DS1_REP8, 'model' : None},
                                 '1k_ds1_denoise_30_r1': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP1, 'model' : None}, 
                                 '1k_ds1_denoise_30_r2': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP2, 'model' : None},
                                 '1k_ds1_denoise_30_r3': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP3, 'model' : None}, 
                                 '1k_ds1_denoise_30_r4': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP4, 'model' : None}, 
                                 '1k_ds1_denoise_30_r5': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP5, 'model' : None}, 
                                 '1k_ds1_denoise_30_r6': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP6, 'model' : None}, 
                                 '1k_ds1_denoise_30_r7': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP7, 'model' : None}, 
                                 '1k_ds1_denoise_30_r8': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_DS1_REP8, 'model' : None}, 
                                 '8k_ds1_denoise_10_r1': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP1, 'model' : None},
                                 '8k_ds1_denoise_10_r2': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP2, 'model' : None},
                                 '8k_ds1_denoise_10_r3': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP3, 'model' : None},
                                 '8k_ds1_denoise_10_r4': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP4, 'model' : None},
                                 '8k_ds1_denoise_10_r5': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP5, 'model' : None},
                                 '8k_ds1_denoise_10_r6': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP6, 'model' : None},
                                 '8k_ds1_denoise_10_r7': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP7, 'model' : None},
                                 '8k_ds1_denoise_10_r8': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_10_8k_b_REP8, 'model' : None},
                                 '8k_ds1_denoise_30_r1': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP1, 'model' : None}, 
                                 '8k_ds1_denoise_30_r2': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP2, 'model' : None},
                                 '8k_ds1_denoise_30_r3': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP3, 'model' : None}, 
                                 '8k_ds1_denoise_30_r4': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP4, 'model' : None}, 
                                 '8k_ds1_denoise_30_r5': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP5, 'model' : None}, 
                                 '8k_ds1_denoise_30_r6': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP6, 'model' : None}, 
                                 '8k_ds1_denoise_30_r7': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP7, 'model' : None}, 
                                 '8k_ds1_denoise_30_r8': {'weights': models.MUNet_v10_Weights.SS_GAUS_DENOISE_v4_30_8k_b_REP8, 'model' : None}, 
                                }

def load_models(device, task = "edge", single_mode_id = None, load_ids = None):
    models_reg = get_model_registry(task = task)
    
    #---- Case 1: single model ----
    if single_mode_id is not None:
        if models_reg[single_mode_id]["model"] is None:
            models_reg[single_mode_id]["model"] = models.munet_v10(weights = models_reg[single_mode_id]["weights"])
            models_reg[single_mode_id]["model"].to(device)
        print(f"Loaded {single_mode_id}")
        return models_reg[single_mode_id]["model"]
    
    models_dict = {}
    
    #--- Case 2: subset of models ---
    if load_ids is not None:
        for model_id in load_ids:
            if models_reg[model_id]["model"] is None:
                models_reg[model_id]["model"] = models.munet_v10(weights = models_reg[model_id]["weights"])
                models_reg[model_id]["model"].to(device)
            print(f"Loaded {model_id}")
            models_dict[model_id] = models_reg[model_id]["model"]
        return models_dict
    
    #--- Case 3: all models -----
    for model_id in models_reg.keys():
        if models_reg[model_id]["model"] is None:
            models_reg[model_id]["model"] = models.munet_v10(weights = models_reg[model_id]["weights"])
            models_reg[model_id]["model"].to(device)
        print(f"Loaded {model_id}")
        models_dict[model_id] = models_reg[model_id]["model"]
    
    return models_dict

def get_model_registry(task="edge"):
    if task == "edge":
        return pretrained_models_edgenets
    elif task == "denoise":
        return pretrained_models_denoisenets
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_model_id(model_id):
    #Expected ids:
    # edge: 1k_ds1_edge_r1 
    #denoise: 1k_ds1_denoise_10_r1
    
    parts = model_id.split("_")
    
    if len(parts) < 4:
        raise ValueError(f"Unexpected model_id format: {model_id}")
    
    size = parts[0]
    ds = parts[1]
    task = parts[2]
    replicate = parts[-1]
    
    
    meta = {
        "model_id": model_id,
        "dataset": f"{size}_{ds}",
        "replicate": replicate,
        "task": task,
    }
    
    if task == "denoise":
        if len(parts) < 5:
            raise ValueError(f"Denoise model_id missing noise level: {model_id}")
        
        noise_std = float(parts[3])/100
        meta["noise_std"] = noise_std
        meta["task"] = "denoise_low" if noise_std == 0.10 else "denoise_high"
        
    return meta

