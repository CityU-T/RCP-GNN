<h1 align="center">
Enhancing Trustworthiness of Graph Neural Networks with Rank-Based Conformal Training 🔥
</h1>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://openreview.net/forum?id=mzGx0j8XYT)
[![](https://img.shields.io/badge/-github-green?style=plastic&logo=github)](https://github.com/CityU-T/RCP-GNN) 
</div>

## Data download
All dataset can be downloaded [here](https://drive.google.com/file/d/1e_wwGGjcw_kDvnpzv3T5tLnbNxjHGdRi/view?usp=drive_link).

## Run RCP-GNN
The hyper-parameters used to train the model is set as default in the `optimal_param_set.pkl` in the training files. Feel free to change them if needed.

Simply run bellow command to reproduce the results in the paper. 

```
python main_smooth.py --model GCN \
                --dataset Cora_ML_CF \
                --device cuda:0 \
                --alpha 0.1\
                --conformal_score thrrank\
                --not_save_res\
                --interpolation higher\
                --num_runs 1\
                --conftr_calib_holdout\
                --conftr\
                --verbose\
```

## Installation
We implement our code by [TrochCP](https://github.com/ml-stat-Sustech/TorchCP) toolbox.

## Reference 
