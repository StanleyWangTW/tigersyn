# TigerSyn
### As python package
**Download**
```
pip install tigersyn
```
**Usage**
```python
import tigersyn
tigersyn.run('s', r'C:\T1w_dir', r'C:\output_dir')
tigersyn.run('s', r'C:\T1w_dir\**\*.nii.gz', r'C:\output_dir')
tigersyn.run('s', r'C:\T1w_dir\**\*.nii.gz') # storing output in the same dir
tigersyn.run('sz', r'C:\T1w_dir') # Force storing nii.gz format
```
```cmd
s: Producing syntheseg mask
z: Force storing nii.gz format
```