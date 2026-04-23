import glob
import re
import os

new_metrics_code = """def compute_metrics(y_true, y_pred):
    import numpy as np
    error = y_pred - y_true
    
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    
    ss_res = np.sum(error**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    medae = np.median(np.abs(error), axis=0)
    max_err = np.max(np.abs(error), axis=0)
    bias = np.mean(error, axis=0)
    acc_10 = np.mean(np.abs(error) <= 10.0, axis=0) * 100.0
    acc_20 = np.mean(np.abs(error) <= 20.0, axis=0) * 100.0

    return {
        "avo_mae_ms": float(mae[0]), "avc_mae_ms": float(mae[1]), "mean_mae_ms": float(mae.mean()),
        "avo_rmse_ms": float(rmse[0]), "avc_rmse_ms": float(rmse[1]), "mean_rmse_ms": float(rmse.mean()),
        "avo_r2": float(r2[0]), "avc_r2": float(r2[1]), "mean_r2": float(r2.mean()),
        "avo_medae_ms": float(medae[0]), "avc_medae_ms": float(medae[1]), "mean_medae_ms": float(medae.mean()),
        "avo_max_err_ms": float(max_err[0]), "avc_max_err_ms": float(max_err[1]), "mean_max_err_ms": float(max_err.mean()),
        "avo_bias_ms": float(bias[0]), "avc_bias_ms": float(bias[1]), "mean_bias_ms": float(bias.mean()),
        "avo_acc_10ms_%": float(acc_10[0]), "avc_acc_10ms_%": float(acc_10[1]), "mean_acc_10ms_%": float(acc_10.mean()),
        "avo_acc_20ms_%": float(acc_20[0]), "avc_acc_20ms_%": float(acc_20[1]), "mean_acc_20ms_%": float(acc_20.mean()),
    }
"""

print_block = """
    print("="*45)
    print(" TEST METRICS ")
    print("="*45)
    print(f"[MAE]  AVO: {metrics['avo_mae_ms']:6.2f} ms | AVC: {metrics['avc_mae_ms']:6.2f} ms | Mean: {metrics['mean_mae_ms']:6.2f} ms")
    print(f"[RMSE] AVO: {metrics['avo_rmse_ms']:6.2f} ms | AVC: {metrics['avc_rmse_ms']:6.2f} ms | Mean: {metrics['mean_rmse_ms']:6.2f} ms")
    print(f"[R²]   AVO: {metrics['avo_r2']:6.3f}    | AVC: {metrics['avc_r2']:6.3f}    | Mean: {metrics['mean_r2']:6.3f}")
    print(f"[MedAE]AVO: {metrics['avo_medae_ms']:6.2f} ms | AVC: {metrics['avc_medae_ms']:6.2f} ms | Mean: {metrics['mean_medae_ms']:6.2f} ms")
    print(f"[MAX]  AVO: {metrics['avo_max_err_ms']:6.2f} ms | AVC: {metrics['avc_max_err_ms']:6.2f} ms | Mean: {metrics['mean_max_err_ms']:6.2f} ms")
    print(f"[BIAS] AVO: {metrics['avo_bias_ms']:6.2f} ms | AVC: {metrics['avc_bias_ms']:6.2f} ms | Mean: {metrics['mean_bias_ms']:6.2f} ms")
    print(f"[<10ms]AVO: {metrics['avo_acc_10ms_%']:6.1f} %  | AVC: {metrics['avc_acc_10ms_%']:6.1f} %  | Mean: {metrics['mean_acc_10ms_%']:6.1f} %")
    print(f"[<20ms]AVO: {metrics['avo_acc_20ms_%']:6.1f} %  | AVC: {metrics['avc_acc_20ms_%']:6.1f} %  | Mean: {metrics['mean_acc_20ms_%']:6.1f} %")
    print("="*45)
"""

files = glob.glob('model/*.py')

for f in files:
    if os.path.basename(f) in ['loso_cv.py']:
        continue
    
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    changed = False
    if 'def compute_metrics' in content:
        pattern = r'def compute_metrics.*?(?=def |class |if __name__|$)'
        # wait, the original replace using `return \{.*?\}` was better to avoid over-replacing.
        pattern2 = r'def compute_metrics\([^)]*\)[^{]*:.*?return \{.*?\}'
        
        content = re.sub(pattern2, new_metrics_code.strip(), content, flags=re.DOTALL)
        changed = True

    # Find the print block (anything matching print.*PEP MAE.*Mean MAE)
    print_pattern = r'print\([^\n]*Test Metrics.*?\n\s*print\([^\n]*Mean RMSE[^\n]*\)'
    if re.search(print_pattern, content, flags=re.DOTALL | re.IGNORECASE):
        content = re.sub(print_pattern, print_block.strip().replace('\n', '\n    '), content, flags=re.DOTALL | re.IGNORECASE)
        changed = True
        
    print_pattern2 = r'print\([^\n]*=== TEST SET METRICS.*?\n\s*print\([^\n]*Mean MAE[^\n]*\)'
    if re.search(print_pattern2, content, flags=re.DOTALL | re.IGNORECASE):
        content = re.sub(print_pattern2, print_block.strip().replace('\n', '\n    '), content, flags=re.DOTALL | re.IGNORECASE)
        changed = True

    if changed:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")
