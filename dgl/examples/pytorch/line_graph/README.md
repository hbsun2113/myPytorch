Community Detection with Graph Neural Networks (CDGNN)
============

Paper link: [https://arxiv.org/abs/1705.08415](https://arxiv.org/abs/1705.08415)

Author's code repo: [https://github.com/joanbruna/GNN_community](https://github.com/joanbruna/GNN_community)

This folder contains a DGL implementation of the CDGNN model.

An experiment on the Stochastic Block Model in default settings can be run with

```bash
python train.py
```

An experiment on the Stochastic Block Model in customized settings can be run with
```bash
python train.py --batch-size BATCH_SIZE --gpu GPU --n-communities N_COMMUNITIES \
                --n-features N_FEATURES --n-graphs N_GRAPH --n-iterations N_ITERATIONS \
                --n-layers N_LAYER --n-nodes N_NODE --model-path MODEL_PATH --radius RADIUS
```
