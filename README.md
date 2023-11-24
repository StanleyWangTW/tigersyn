# TigerSyn
### As python package
**Download**
```
pip install https://github.com/StanleyWangTW/tigersyn/archive/release.zip
```
**Usage**
```python
import tigersyn
tigersyn.run('s', r'C:\T1w_dir', r'C:\output_dir')
tigersyn.run('s', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')
tigersyn.run('s', r'C:\T1w_dir\**\*.nii.gz') # storing output in the same dir
tigersyn.run('sz', r'C:\T1w_dir') # Force storing nii.gz format
```
```
s: Producing SynthSeg mask
h: Producing hippocampus mask
z: Force storing nii.gz format
```

**Syntheseg labels**
| Label | Structure              | Label | Structure               |
| ----- | ---------------------- | ----- | ----------------------- |
| 2     | Left Cerebral WM       | 41    | Right Cerebral WM       |
| 3     | Left Cerebral Cortex   | 42    | Right Cerebral Cortex   |
| 4     | Left Lateral Ventricle | 43    | Right Lateral Ventricle |
| 5     | Left Inf Lat Vent      | 44    | Right Inf Lat Vent      |
| 7     | Left Cerebellum WM     | 46    | Right Cerebellum WM     |
| 8     | Left Cerebellum Cortex | 47    | Right Cerebellum Cortex |
| 10    | Left Thalamus          | 49    | Right Thalamus          |
| 11    | Left Caudate           | 50    | Right Caudate           |
| 12    | Left Putamen           | 51    | Right Putamen           |
| 13    | Left Pallidum          | 52    | Right Pallidum          |
| 14    | 3rd Ventricle          | 53    | Right Hippocampus       |
| 15    | 4th Ventricle          | 54    | Right Amygdala          |
| 16    | Brain Stem             | 58    | Right Accumbens area    |
| 17    | Left Hippocampus       | 60    | Right VentralDC         |
| 18    | Left Amygdala          |
| 26    | Left Accumbens area    |
| 28    | Left VentralDC         |