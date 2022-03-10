<!-- # DeCLIP
Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm.

Our paper is available on [arxiv](https://arxiv.org/abs/2110.05208) -->


# [Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm.](https://arxiv.org/abs/2110.05208)

DeCLIP: Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm. Our paper is available on [Arxiv](https://arxiv.org/abs/2110.05208).

DeCLIP is an open-source project that welcomes any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as a standardized toolkit to reimplement existing methods and develop their own new Contrastive Language-Image Pretraining methods. You can find the following things in this repo:
+ Pre-trained models and training codes to reproduce various Contrastive Language-Image Pretraining methods(e.g. CLIP, DeCLIP, SLIP, FILIP).
+ Various benchmark datasets for Large-scale Contrastive Language-Image Pretraining task.
+ Zero-shot transfer and linear classification evaluation scripts for downstream datasets.

## Introduction

Recently, large-scale Contrastive Language-Image Pre-training (CLIP) (Radfordet al., 2021) has attracted unprecedented attention for its impressive zero-shot recognition ability and excellent transferability to downstream tasks. However, CLIP is quite data-hungry and requires 400M image-text pairs for pre-training, thereby restricting its adoption. This work proposes a novel training paradigm, Data efficient CLIP (DeCLIP), to alleviate this limitation. We demonstrate that by carefully utilizing the widespread supervision among the image-text pairs, our DeCLIP can learn generic visual features more efficiently. Instead of using the single image-text contrastive supervision, we fully exploit data potential through the use of (1) self-supervision within each modality; (2) multi-view supervision across modalities; (3) nearest-neighbor supervision from other similar pairs. Benefiting from these intrinsic supervision, our DeCLIP-ResNet50 can achieve 60.4% zero-shot top1 accuracy on ImageNet, which is 0.8% above the CLIP-ResNet50 while using 7.1× fewer data. Our DeCLIP-ResNet50 outperforms its counterpart in 8 out of 11 visual datasets when transferred to downstream tasks. Moreover, Scaling up the model and computing also works well in our framework.


<p align="center"><img src="docs/main_figure.jpg" alt="Declip framework" width="800"/></p>

<!-- ![main_figure](docs/main_figure.jpg) -->



# Updates

***2022-03-10*** We update the result of CLIP-Benchmark and release our YFCC15M

***2022-02-22*** We release our training code, benchmark, and model zoo! ***We will release the checkpoints of each models after align the results soon***. We hope this project could serve the growing Contrastive Language-Image Pretraining research community by providing a flexible as well as standardized toolkit.

***2021-11-06*** First Commit, Our code, dataset and models will be relased soon.


## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.


## Get Started


## CLIP-Benchmark

<!-- **Model will be relased soon** -->

### Supported Models:

The following models are pre-trained on YFCC15M and evaluated on ImageNet-1K (ILSVRC2012).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Dataset</th>
<th valign="center">Model</th>
<th valign="center">Epochs</th>
<th valign="center">0-shot</th>
<th valign="center">Config</th>
<th valign="center">Paper</th>
<th valign="center">Weights</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ViT-B32</td>
<td align="center">32</td>
<td align="center">32.8</td>
<td align="center"><a href="experiments/clip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2103.00020.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">DeCLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ViT-B32</td>
<td align="center">32</td>
<td align="center">43.2</td>
<td align="center"><a href="experiments/declip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2110.05208.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">SLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ViT-B32</td>
<td align="center">32</td>
<td align="center">34.3</td>
<td align="center"><a href="experiments/slip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2112.12750.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">FILIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ViT-B32</td>
<td align="center">32</td>
<td align="center">39.5</td>
<td align="center"><a href="experiments/filip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2111.07783.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>

<td align="center">DeFILIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ViT-B32</td>
<td align="center">32</td>
<td align="center">45.0</td>
<td align="center"><a href="experiments/defilip_experiments">config</a></td>
<td align="center"><a href="">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>

</tbody></table>




<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Dataset</th>
<th valign="center">Model</th>
<th valign="center">Epochs</th>
<th valign="center">0-shot</th>
<th valign="center">Config</th>
<th valign="center">Paper</th>
<th valign="center">Weights</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ResNet50</td>
<td align="center">32</td>
<td align="center">37.2</td>
<td align="center"><a href="experiments/clip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2103.00020.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">DeCLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ResNet50</td>
<td align="center">32</td>
<td align="center">44.4</td>
<td align="center"><a href="experiments/declip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2110.05208.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">SLIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ResNet50</td>
<td align="center">32</td>
<td align="center">28.5</td>
<td align="center"><a href="experiments/slip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2112.12750.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
<tr>
<td align="center">FILIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ResNet50</td>
<td align="center">32</td>
<td align="center">21.3</td>
<td align="center"><a href="experiments/filip_experiments">config</a></td>
<td align="center"><a href="https://arxiv.org/pdf/2111.07783.pdf">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>

<!-- <td align="center">DeFILIP</td>
<td align="center">YFCC-15M</td>
<td align="center">ResNet50</td>
<td align="center">32</td>
<td align="center">--</td>
<td align="center"><a href="experiments/defilip_experiments">config</a></td>
<td align="center"><a href="">paper</a></td>
<td align="center"><a href="">url</a></td>
</tr>
 -->
</tbody></table>

### Supported datasets:


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Samples</th>
<th valign="center">download</th>
<th valign="center">Paper</th>

<!-- TABLE BODY -->
<tr>
<td align="center">YFCC-15M</td>
<td align="center">15,388,848</td>
<td align="center"><a href="https://drive.google.com/file/d/1P-2_dHNc_c5XMY0A-89iNF5Cz_Y_Cfsy/view?usp=sharing">google driver</a></td>
<td align="center"><a href="">url</a></td>
</tr>

</tbody></table>


<!-- ### Our pretrain visual backbone model (w/o text encoder)

DeCLIP_r50    [GoogleDriver](https://drive.google.com/file/d/1SZJ8CU5dDIwuvZWxb4xdld7qv7aw6wKm/view?usp=sharing).  
DeCLIP_vitb32 [GoogleDriver](https://drive.google.com/file/d/1W2cCxsr3EjvOOWzVXZukLk38c8LC6UUm/view?usp=sharing) -->

<!-- 
### Our pretrain visual backbone 

**Model will be relased soon**  -->



## Changelog

***2022-02-22*** Realase our Training code

***2021-11-06*** First Commit
 


## Citation

```
@misc{li2021supervision,
      title={Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm}, 
      author={Yangguang Li and Feng Liang and Lichen Zhao and Yufeng Cui and Wanli Ouyang and Jing Shao and Fengwei Yu and Junjie Yan},
      year={2021},
      eprint={2110.05208},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License. For commercial use, please contact the authors.

## Acknowledgement

DeCLIP is an open-source project that welcomes any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as a standardized toolkit to reimplement existing methods and develop their own new Contrastive Language-Image Pretraining methods.

Our framework is based on [prototype](https://github.com/ModelTC/prototype).


