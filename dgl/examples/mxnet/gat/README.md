Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo:
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).

Note that the original code is implemented with Tensorflow for the paper.


## Usage (make sure that DGLBACKEND is changed into mxnet)
```bash
DGLBACKEND=mxnet python gat_batch.py --dataset cora --gpu 0 --num-heads 8
```
