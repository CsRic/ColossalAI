import colossalai
import torch
from colossalai.nn.parallel.layers.cache_embedding.freq_aware_embedding import FreqAwareEmbeddingBag
from colossalai.nn.parallel.layers.cache_embedding.cache_mgr import EvictionStrategy
import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
Bag = FreqAwareEmbeddingBag(
    5,
    5,
    cuda_row_num=2,
    buffer_size=0,
    pin_weight=True,
    warmup_ratio=0.0,
    ids_freq_mapping=torch.tensor([5,4,3,2,1],device="cuda:0"),
    evict_strategy=EvictionStrategy.LFU
)

offsets = torch.tensor([0],device="cuda:0")

Bag.forward(torch.tensor([0,1],device="cuda:0"),offsets)
Bag.forward(torch.tensor([1,2],device="cuda:0"),offsets)
Bag.forward(torch.tensor([4,0],device="cuda:0"),offsets)
Bag.forward(torch.tensor([2,3],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,1],device="cuda:0"),offsets)
Bag.forward(torch.tensor([0,1],device="cuda:0"),offsets)

print(Bag.cache_weight_mgr.num_hits_history)
print(Bag.cache_weight_mgr.num_miss_history)