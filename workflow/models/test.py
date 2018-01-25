import torch
import torch.nn as nn
import densenet_efficient as dens
fcn = dens.DenseNetClassification(growth_rate=12, block_config=(4, 4, 4, 4), compression=0.5,
                                                     num_init_features=24, bn_size=4, drop_rate=0)
