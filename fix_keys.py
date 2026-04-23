import glob

files = [
    'model/train_cnn_dual_advanced.py',
    'model/train_cnn_dual_branch.py',
    'model/train_cnn_dual_smooth_clip.py'
]

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    content = content.replace('pep_mae_ms', 'avo_mae_ms')
    content = content.replace('pep_rmse_ms', 'avo_rmse_ms')
    content = content.replace('val_pep_mae', 'val_avo_mae')
    content = content.replace('PEP MAE', 'AVO MAE')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f'Fixed keys in {f}')
