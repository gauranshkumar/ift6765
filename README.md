---
dataset_info:
  features:
  - name: sketch_image
    dtype: image
  - name: tikz_code
    dtype: string
  - name: tool
    dtype: string
  - name: uri
    dtype: string
  splits:
  - name: train
    num_bytes: 1582896241.72
    num_examples: 2585
  - name: validation
    num_bytes: 186882956.0
    num_examples: 323
  - name: test
    num_bytes: 185203331.0
    num_examples: 323
  download_size: 1835913422
  dataset_size: 1954982528.72
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---
