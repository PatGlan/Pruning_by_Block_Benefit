# Depth Aware Pruning




 Abstract:
>Vision Transformer have set new benchmarks in several tasks, but these models come with the lack of high computational costs which makes them impractical for resource limited hardware. Network pruning reduces the computational complexity by removing less important operations while maintaining performance. However, when a pretrained model is pruned before convergence in an unseen data domain, it can lead to a misevaluation of weight significance, resulting in an unfavourable resource assignment. To face the issue of removing elements too early in training, we propose Depth Aware Pruning (DAP), which assigns model parameters dependent on the state of convergence. Our proposed method identifies the changing impact of task specific features and rebalances the computational resources to boost lately converged components. 

---

Pruning on transfer learning tasks causes a domain mismatch between the pretrained model initialization and the target domain of the downstream task.
This domain gap leads to a misevaluation of weight importance, resulting in performance loss when model parameters are removed to early.
Our experiments point out deeper layers express task specific features in later epochs, shown in the following figure.
We findings that the training time at which layers adapt to a new domain depends on the network depth, leading to a structural problem in pruning.

<p align="center">
<img src="fig/BlockPerformance_overTrainingEpochs.png"  width="400" height="300">
</p>
> We measure the discriminative feature improvement of individual Attention and MLP blocks for classification token (upper row) and patches (botton row) over training epochs.
> The results show, deeper layers express features only in later epochs, while shallower layers stay almost equal discriminative ovet the whole training period.


To face this problem we porpose *Depth Aware Pruning* (*DAP*) to balance the global parameter resources dependent on the feature improvement of Attention and MLP blocks.
Thereby DAP ensures high parameter resources to be assiged, only if the regarding block contributes to the featureimprovement.
As shown in the following table, DAP in highly performant on transfer learning tasks, by considering the depth dependent state of convergence of individual blocks.


 | model      | method         | pruned  | IFOOD <br> pr=50%   | IFOOD <br> pr=75%   | INAT19 <br> pr=50%  | INAT19 <br> pr=75%  |
 |:-----------|:---------------|:--------|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
 | Deit-small | Deit           | &cross; | 73.9                                     || 74.7                                     ||
 |            | WD-Prune       | &check; | 50.7                | 49.2                | 55.6                | 54.0                |
 |            | SaVit          | &check; | 72.4                | 64.4                | 71.3                | 68.0                |
 |            | **DAP (ours)** | &check; | **74.3**            | **73.4**            | **75.5**            | **73.1**            |
 | Deit-tiny  | Deit           | &cross; | 72.7                                     || 72.6                                     ||
 |            | WD-Prune       | &check; | 50.2                | 44.7                | 54.8                | 46.7                |
 |            | SaVit          | &check; | 65.7                | 59.5                | 64.1                | 45.3                |
 |            | **DAP (ours)** | &check; | **71.5**            | **68.6**            | **69.3**            | **61.4**            |

<br>

---

## Install
checkout our repository
```
git clone https://anonymous.4open.science/r/DepthAwarePruning-DE53
cd DepthAwarePruning

```

Use the provided `requirements.txt` file to create a conda environment for this project: 
```
conda create --name DAP --file requirements.txt
```


---

## Run
