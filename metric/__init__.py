from .classification import FG_Classification, CG_Classification
from .vqa import VQA, MMBenchVQA
from .caption import Caption
from .desiderata import MMBench_Calibration, ScienceQA_Calibration, POPE_Metric, Instruct_Follow
from .detection import Detection, KOSMOS_Detection 
from .counting import Counting
# from .pope import POPE_Metric

evaluation_protocol = {
    'basic':{
        'CIFAR10': CG_Classification,
        'Omnibenchmark': FG_Classification,
        'Flickr30k' : Caption,
        'ScienceQA': VQA,
        'VOC2012': Detection,
        'FSC147': Counting,
        'MMBench': MMBenchVQA
    },
    # 'MMBench':{
    #     'VQA': MMBench,
    # },
    'Calibration':
    {
        'VQA': ScienceQA_Calibration,
        'MMBench': MMBench_Calibration
    },
    # 'MMBench_Calibration':
    # {
    #     'VQA': MMBench_Calibration,
    # },
    'Hallucination':
    {
        'POPE': POPE_Metric,
    },
    'Instruct_Follow':
    {
      'VQA': Instruct_Follow,
      'MMBench': Instruct_Follow,
    },
    # 'Consistency':
    # {
    #     'VQA': LAMM_VQA_CON
    # },
    'KOSMOS':{
        'VOC2012': KOSMOS_Detection,
    },  
    # 'MME':{
    #     'VQA': MME,
    # }

}

def build_metric(metric_type, dataset_name, **kwargs):
    build_fuc = evaluation_protocol[metric_type][dataset_name]
    return build_fuc(dataset_name = dataset_name, **kwargs)