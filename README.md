<div align="center"> 
    <h1> Pruning by Block Benefit: Exploring the Properties of Vision Transformer
Blocks during Domain Adaptation </h1>
</div>

<div align="center"> 
<a href="https://arxiv.org/abs/2506.23675">
  <img src="https://img.shields.io/badge/ArXiv-2506.23675-red?style=flat&label=ArXiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F2506.23675" alt="Static Badge" />
</a>
<a href="https://openreview.net/forum?id=NjzZunViBb">
  <img src="https://img.shields.io/badge/ICML-OpenReview-blue?style=flat&label=ICML&link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DNjzZunViBb" alt="Static Badge" />
</a>
<a href="https://patglan.github.io/Pruning_by_Block_Benefit/">
  <img src="https://img.shields.io/badge/Project_Page-green?style=flat&label=Github.io&link=https%3A%2F%2Fpatglan.github.io%2FPruning_by_Block_Benefit%2F" alt="Static Badge" />
</a>
<a href="https://github.com/PatGlan/Pruning_by_Block_Benefit">
  <img src="https://img.shields.io/badge/GitHub-Code-yellow?style=flat&link=https%3A%2F%2Fgithub.com%2FPatGlan%2FPruning_by_Block_Benefit" alt="Static Badge" />
</a>
</div>

---

# Pruning by Block Benefit (P3B)


This GitHub repository contains the code for our paper: **Pruning by Block Benefit: Exploring the Properties of Vision Transformer Blocks during Domain Adaptation**

 Abstract:
>Vision Transformer have set new benchmarks in several tasks, but these models come with the lack of high computational costs which makes them impractical for resource limited hardware. Network pruning reduces the computational complexity by removing less important operations while maintaining performance. However, pruning a pretrained model on an unseen data domain, leads to a misevaluation of weight significance, resulting in an unfavourable resource assignment. To address the issue of efficient pruning on transfer learning tasks, we propose Pruning by Block Benefit (P3B) , a method to globally assign a given parameter budget depending on the relative contribution of individual blocks. Our proposed method is able to identify lately converged components in order to rebalance the global parameter resources. Furthermore, our findings show that the order in which layers express valuable features on unseen data depends on the network depth, leading to a structural problem in pruning.
---


# Motivation

Common pruning strategies often fail in transfer learning because early pruning misevaluates weight significance before the model converges to the target domain. 
Concretely, the early removement of prunable elements leads to a misevaluation of weight significance, since task sensitive elements become important, only once the model has been converged to the target domain.
This premature parameter elimination compromises the resulting model structure and harms performance.

Our work points out an overlooked aspect in pruning, regarding the model depth.
As visualized in the figure below, our work shows that deeper layers converge later in training, harming early pruning decisions.
Therefore, P3B establishes block-specific pruning rates based on the relative performance of each block. 
This approach effectively identifies later-converging blocks and guarantees their reactivation during training.

<p align="center">
<img src="fig/BlockPerformance_overTrainingEpochs.png"  width="400" height="400">
</p>
<p align="center" style="font-size: 0.9em; color: gray;">
  <b>Figure 1:</b> The relative feature improvement on classification token (upper row) and patches (botton row) for individual Attention and MLP blocks is depth dependent. Deeper layers express features only in later epochs.
</p>

---

# Method

In this work we propose the novel pruning framwork **Pruning by Block Benefit (P3B)** to balance the global parameter resources dependent on the feature improvement of Attention and MLP blocks.
The approach is designed to decouple the inter-block pruning ratio from the intra-layer element elimination, ensuring their respective criteria are assessed independently.
As illustrated in Figure 2, **P3B** determines a block-wise parameter budget by utilizing the relative feature improvement demonstrated Block Performacne Indicator $\Delta\Psi_i$. 
This defined layer-wise sparsity constraint then guides the generation of the pruning mask via local pruning criteria.
*P3B* is a highly performant pruning framework, considering the structural change of the model during domain adaptaion.

<p align="center">
<img src="fig/Block_plan.png"  width="600" height="400">
</p>
<p align="center" style="font-size: 0.9em; color: gray;">
  <b>Figure 2:</b> Block plan P3B. We measure the realtive performance of each block to set a superior parameter budet per block. 
</p>

As shown in the following table, *P3B* significantly outperforms existing pruning methods.

    
 | model      | method         | pruned  | IFOOD <br> pr=50%   | IFOOD <br> pr=75%   | INAT19 <br> pr=50%  | INAT19 <br> pr=75%  |
 |:-----------|:---------------|:--------|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
 | Deit-Small | Deit           | &cross; | 73.9                                     || 74.7                                     ||
 |            | WD-Prune       | &check; | 50.7                | 49.2                | 55.6                | 54.0                |
 |            | SaVit          | &check; | 72.4                | 64.4                | 71.3                | 68.0                |
 |            | **P3B (ours)** | &check; | **74.3**            | **73.4**            | **75.5**            | **73.1**            |
 | Deit-Tiny  | Deit           | &cross; | 72.7                                     || 72.6                                     ||
 |            | WD-Prune       | &check; | 50.2                | 44.7                | 54.8                | 46.7                |
 |            | SaVit          | &check; | 65.7                | 59.5                | 64.1                | 45.3                |
 |            | **P3B (ours)** | &check; | **71.5**            | **68.6**            | **69.3**            | **61.4**            |
 
<br>

---

## Install
<checkout repository>
checkout our repository

```
git clone https://anonymous.4open.science/r/Pruning_by_Block_Benefit-0416
cd Pruning_by_Block_Benefit
```

<install environment>
Use the provided `requirements.txt` file to create a conda environment for this project: 

```
conda create --name P3B --file requirements.txt
```


---

## Run

prune model with P3B
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port $USED_PORT --use_env main.py --output_dir $OUTPUT_DIR --epochs 50 --batch-size 64 --grad-accum-steps 2 --data-set CIFAR100 --data-path $PATH_DATASET --model deit_small_patch16_224 --model-path $MODEL_PATH --prune-method "block_pr" --prune-ratio $PRUNE_RATIO --prune-steps-update-mask 100 --prune-start-epoch 10 --dist-eval
```
*Note: we scale argument "prune-steps-update-mask" to 3 mask update steps per epoch.*

fine-tune pruned model
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port $USED_PORT --use_env main.py  --output_dir $OUTPUT_DIR --batch-size 128 --data-set CIFAR100 --data-path $PATH_DATASET --model deit_small_patch16_224 --model-path ${OUTPUT_DIR}/first_sparse_checkpoint.pth --resize-model-by-checkpoint --dist-eval
```

# Citation


If you use this code in your research, please cite the following paper:

```
@inproceedings{
  glandorf2025p3b,
  title={Pruning by Block Benefit: Exploring the Properties of Vision Transformer Blocks during Domain Adaptation},
  author={Patrick Glandorf and Bodo Rosenhahn},
  booktitle={International Conference on Computer Vision Workshop},
  year={2025}
}
```






