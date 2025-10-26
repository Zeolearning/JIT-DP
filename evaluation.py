from pathlib import Path
from util.util import CONSTANTS
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
def eval_res():
    res_path=Path(CONSTANTS.repository_dir)/"predictions.csv"
    data=pd.read_csv(res_path)
    ground_truth=data['is_buggy_commit']
    DSR_prediction=data['predicted_result']
    f1 = f1_score(ground_truth, DSR_prediction)
    auc = roc_auc_score(ground_truth, DSR_prediction)
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")


eval_res()