# Model Examples using DGL (w/ Pytorch backend)

Each model is hosted in their own folders. Please read their README.md to see how to
run them.

To understand step-by-step how these models are implemented in DGL. Check out our
[tutorials](https://docs.dgl.ai/tutorials/models/index.html)

## Model summary

Here is a summary of the model accuracy and training speed. Our testbed is Amazon EC2 p3.2x instance (w/ V100 GPU).

| Model | Reported <br> Accuracy | DGL <br> Accuracy | Author's training speed (epoch time) | DGL speed (epoch time) | Improvement |
| ----- | ----------------- | ------------ | ------------------------------------ | ---------------------- | ----------- |
| [GCN](https://arxiv.org/abs/1609.02907)   | 81.5% | 81.0% | [0.0051s (TF)](https://github.com/tkipf/gcn) | 0.0042s | 1.17x |
| [TreeLSTM](http://arxiv.org/abs/1503.00075) | 51.0% | 51.72% | [14.02s (DyNet)](https://github.com/clab/dynet/tree/master/examples/treelstm) | 3.18s | 4.3x |
| [R-GCN <br> (classification)](https://arxiv.org/abs/1703.06103) | 73.23% | 73.53% | [0.2853s (Theano)](https://github.com/tkipf/relational-gcn) | 0.0273s | 10.4x |
| [R-GCN <br> (link prediction)](https://arxiv.org/abs/1703.06103) | 0.158 | 0.151 | [2.204s (TF)](https://github.com/MichSchli/RelationPrediction) | 0.633s | 3.5x |
| [JTNN](https://arxiv.org/abs/1802.04364) | 96.44% | 96.44% | [1826s (Pytorch)](https://github.com/wengong-jin/icml18-jtnn) | 743s | 2.5x |
| [LGNN](https://arxiv.org/abs/1705.08415) | 94% | 94% | n/a | 1.45s | n/a |
| [DGMG](https://arxiv.org/pdf/1803.03324.pdf) | 84% | 90% | n/a | 238s | n/a |
