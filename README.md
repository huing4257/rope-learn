# Rotary Encoding learning

## usage

``` bash
python setup.py build_ext --inplace
```

```python
import torch
import rope
q = torch.randn(8,1,64).cuda() # NHD layout
q = rope.ops.rotary_pos_encoding(q)
```
