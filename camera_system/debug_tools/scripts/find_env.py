import subprocess
import sys

# 找有 numpy 的 python3
candidates = [
    '/home/yp8700/miniconda3/bin/python3',
    '/home/yp8700/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3',
    '/usr/bin/python3',
]

for py in candidates:
    result = subprocess.run([py, '-c', 'import numpy, cv2; print(py, numpy.__version__, cv2.__version__)'.replace('py', repr(py))], 
                            capture_output=True, text=True)
    print(f"{py}: {result.stdout.strip() or result.stderr.strip()[:60]}")
