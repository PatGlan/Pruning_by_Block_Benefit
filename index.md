<div align="center"> 
    <h1> Pruning by Block Benefit: Exploring the Properties of Vision Transformer Blocks during Domain Adaptation </h1>
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


<p style="font-style: italic; background-color: #f0f0f0; padding: 10px; display: inline-block; text-align: justify;">
Vision Transformer have set new benchmarks in several tasks, but these models come with the lack of high computational costs which makes them impractical for resource limited hardware. Network pruning reduces the computational complexity by removing less important operations while maintaining performance. However, pruning a pretrained model on an unseen data domain, leads to a misevaluation of weight significance, resulting in an unfavourable resource assignment. To address the issue of efficient pruning on transfer learning tasks, we propose Pruning by Block Benefit (P3B) , a method to globally assign a given parameter budget depending on the relative contribution of individual blocks. Our proposed method is able to identify lately converged components in order to rebalance the global parameter resources. Furthermore, our findings show that the order in which layers express valuable features on unseen data depends on the network depth, leading to a structural problem in pruning.
</p>

This GitHub repository contains the code for our paper: **Pruning by Block Benefit: Exploring the Properties of Vision Transformer Blocks during Domain Adaptation**

---


<p align="center">
<img src="fig/BlockPerformance_overTrainingEpochs.png"  width="400" height="300">
</p>
The relative feature improvement on classification token (upper row) and patches (botton row) for individual Attention and MLP blocks is depth dependent.
Deeper layers express features only in later epochs.

When applying pruning strategies to transfer learning tasks, the discrepancy between the initial model domain and the target domain of the downstream task must be considered.
Neglecting this factor leads to unfavourable pruned weights, ultimately causing performance degradation when model parameters are eliminated prematurely.
Moreover, our work points out an overlooked aspect in pruning, regarding the model depth.
As visualized in the figure above, deeper layers converge later in training, harming early pruning decisions.
This raises the question: "When to prune individual layers"?

We propose the novel pruning framwork **Pruning by Block Benefit (P3B)** to balance the global parameter resources dependent on the feature improvement of Attention and MLP blocks.
*P3B* is a highly performant pruning framework, considering the structural change of the model during domain adaptaion.
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

