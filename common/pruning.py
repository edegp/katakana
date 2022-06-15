import numpy as np

def create_layer_mask(weights, pruning_rate):
    # print(weights.shape)
    abs_param = abs(weights)
    # print(abs_param)
    data = sorted(np.ndarray.flatten(abs_param))
    num_prune = int(len(data) * pruning_rate)
    idx_prune = min(num_prune, len(data)-1)
    threshould = data[idx_prune]
    # print(threshould)
    mask = abs_param
    mask[mask < threshould] = 0
    mask[mask >= threshould] = 1
    return mask
def create_model_mask(params, pruning_rate):
    masks = {}
    for key, val in params.items():
        mask = create_layer_mask(val, pruning_rate)
        masks[key] = mask
    return masks
def prune_weight(params, masks):
        for key in params.keys():
            if key not in masks.keys():
                continue
            mask = masks[key]
            params[key] = params[key] * mask
