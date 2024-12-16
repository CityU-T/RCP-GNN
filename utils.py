import numpy as np
from torchcp.classification import Metrics


def conf_pred(predictor, smx, n_base, labels, alpha, interpolation, num_classes = None):
    
    cov_all = []
    eff_all = []

    for k in range(100):    
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0  # n_base = n is the modified calib set size
        # print("idx: " + str(idx))
        np.random.seed(k)
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        # print("cal_smx: " + str(cal_smx))
        cal_labels, val_labels = labels[idx], labels[~idx]
        predictor.calculate_threshold(cal_smx, cal_labels, alpha, interpolation)
        prediction_sets = predictor.predict_with_logits(val_smx)

        # metrics = Metrics()
        result_dict = predictor.evaluate(prediction_sets, val_labels) 

        cov_all.append(result_dict["Coverage_rate"])
        eff_all.append(result_dict["Average_size"])

    return np.mean(cov_all), np.mean(eff_all) #