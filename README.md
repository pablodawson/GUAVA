---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Library: [More Information Needed]
- Docs: [More Information Needed]

---
license: apache-2.0
- en
---
<div align="center">
<h1>LAM: Large Avatar Model for One-shot Animatable Gaussian Head</h1>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/pdf/2502.17796'><img src='https://img.shields.io/badge/📜-arXiv:2503-10625'></a> 
  <a href='https://aigc3d.github.io/projects/LAM/'><img src='https://img.shields.io/badge/🌐-Project_Website-blueviolet'></a> 
  <a href='https://huggingface.co/spaces/3DAIGC/LAM'><img src='https://img.shields.io/badge/🤗-HuggingFace_Space-blue'></a> 
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/📃-Apache--2.0-929292"></a>
</div>
</div>


## Overview

This repository contains the example data and face blendshape model of the paper [LAM](https://arxiv.org/pdf/2502.17796). 

LAM creates animatable Gaussian heads with one-shot images in a single forward pass in seconds. The reconstructed Gaussian avatar can
be reenacted and rendered on various platforms in real-time.


## Quick Start

Please refer to our [Github Repo](https://github.com/aigc3d/LAM)

### Download Model
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="3DAIGC/LAM-assets",
                repo_type='model',
                filename='LAM_human_model.tar',
                local_dir='./')
os.system('tar -xf LAM_human_model.tar && rm LAM_human_model.tar')

# launch example for LAM
hf_hub_download(repo_id='3DAIGC/LAM-assets',
                repo_type='model',
                filename='LAM_assets.tar',
                local_dir='./')
os.system('tar -xf LAM_assets.tar && rm LAM_assets.tar')
```


## Citation 
```
@article{he2025lam,
  title={LAM: Large Avatar Model for One-shot Animatable Gaussian Head},
  author={He, Yisheng and Gu, Xiaodong and Ye, Xiaodan and Xu, Chao and Zhao, Zhengyi and Dong, Yuan and Yuan, Weihao and Dong, Zilong and Bo, Liefeng},
  journal={arXiv preprint arXiv:2502.17796},
  year={2025}
}
```
