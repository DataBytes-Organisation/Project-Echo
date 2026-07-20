import os, sys, traceback
root = os.path.abspath(os.getcwd())
files = [
    'src/Components/projected_movement.py',
    'src/Components/movement_prediction.py',
    'src/Components/clean_movement_data.py',
    'src/io/export_data.py',
    'src/io/export_combined_data.py',
    'src/Components/Simulator/src/comms_manager.py',
    'src/Components/vegetation_density.py'
]

ok = True
for f in files:
    p = os.path.join(root, f)
    if not os.path.exists(p):
        print(f'MISSING: {f}')
        ok = False
        continue
    try:
        src = open(p, 'r', encoding='utf-8').read()
        compile(src, p, 'exec')
        print(f'OK: {f}')
    except Exception:
        ok = False
        print(f'ERROR in {f}:')
        traceback.print_exc()

if not ok:
    sys.exit(2)
print('SMOKE TESTS PASSED')
