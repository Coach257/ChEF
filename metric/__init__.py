from .classification import FG_Classification, CG_Classification
from .desiderata import MMBench_Calibration, ScienceQA_Calibration
from .vqa import VQA
# from .caption import  LAMM_Caption
# from .vqa import  LAMM_VQA, MMBench, LAMM_VQA_Calibration, MMBench_Calibration, LAMM_VQA_INF,LAMM_VQA_CON,MME
from .detection import Detection, KOSMOS_Detection 
# from .counting import LAMM_Counting
# from .pope import POPE_Metric

evaluation_protocol = {
    'basic':{
        'coarse_grained_classification': CG_Classification,
        'fine_grained_classification': FG_Classification,
        # 'caption' : LAMM_Caption,
        'VQA': VQA,
        'detection': Detection,
        
        # 'counting': LAMM_Counting,
    },
    # 'MMBench':{
    #     'VQA': MMBench,
    # },
    'Calibration':
    {
        'ScienceQA': ScienceQA_Calibration,
        'MMBench': MMBench_Calibration
    },
    # 'MMBench_Calibration':
    # {
    #     'VQA': MMBench_Calibration,
    # },
    # 'Hallucination':
    # {
    #     'POPE': POPE_Metric,
    # },
    # 'Instruct_Follow':
    # {
    #     'VQA': LAMM_VQA_INF,
    # },
    # 'Consistency':
    # {
    #     'VQA': LAMM_VQA_CON
    # },
    'KOSMOS':{
        'detection': KOSMOS_Detection,
    },  
    # 'MME':{
    #     'VQA': MME,
    # }

}

def build_metric(metric_type, task_name, dataset_name, **kwargs):
    build_fuc = evaluation_protocol[metric_type][task_name]
    return build_fuc(dataset_name = dataset_name, **kwargs)