# This is the code implemention for RCP-GNN

You can start it with below command:

'''json
python main_smooth.py --model GCN \
                --dataset Cora_ML_CF \
                --device cuda:0 \
                --alpha 0.1\
                --conformal_score thrrank\
                --not_save_res\
                --interpolation higher\
                --conf_epochs 5000\
                --num_runs 1\
                --conftr_calib_holdout\
                --conftr\
'''
