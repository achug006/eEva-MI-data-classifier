import json
with open('Author_EDA.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
targets = ['#2ecc71','#e74c3c','#95a5a6','#ff9999','steelblue','YlOrRd','viridis']
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        for t in targets:
            if t in src:
                print(f'Cell {i+1}: {t}')
