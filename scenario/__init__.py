from .caption_dataset import *
from .classification import *
from .vqa_dataset import *
from .det_dataset import *
from .counting_dataset import *
from .MMBench_dataset import *
from .POPE_dataset import *

from .SEED_Bench_dataset import *
from .MME_dataset import *
dataset_dict = {
    # Caption 
    'Flickr': FlickrDataset,
    # classification
    'CIFAR10': CIFAR10Dataset,
    'Omnibenchmark': OmnibenchmarkDataset,
    # VQA 
    'ScienceQA': ScienceQADataset,
    # Detection
    'VOC2012': VOC2012Dataset,
    # Counting
    'FSC147': FSC147Dataset,
    # MMBench
    'MMBench': MMBenchDataset,
    #'MMBench_align': MMBenchDataset_aligned,
    'POPE_COCO_random':POPE_COCO_Random_Dataset,
    'POPE_COCO_popular':POPE_COCO_Popular_Dataset,
    'POPE_COCO_adversarial':POPE_COCO_Adversarial_Dataset,
    # SEEDBench
    'SEEDBench': SEEDBenchDataset,
    'MME': MMEDataset,
}
