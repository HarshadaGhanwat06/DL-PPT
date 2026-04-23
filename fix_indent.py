import glob

files = glob.glob('model/*.py')
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    fixed_lines = []
    for line in lines:
        if 'print("="*45)' in line or 'print(" TEST METRICS ")' in line or 'print(f"[' in line:
            # We want precisely 4 spaces of indentation
            line = '    ' + line.lstrip()
        fixed_lines.append(line)
        
    with open(f, 'w', encoding='utf-8') as file:
        file.writelines(fixed_lines)
