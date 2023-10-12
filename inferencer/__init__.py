from .Direct import Direct_inferencer
from .PPL import PPL_inferencer, ICL_PPL_inferencer

inferencer_dict = {
    'Direct': Direct_inferencer,
    # 'detection': Direct_inferencer,
    # 'detection_ppl': Det_PPL_inferencer,
    'PPL': PPL_inferencer,
    # 'multi_ppl': Multi_Turn_PPL_inferencer,
    # 'In_context_learning': Icl_inferencer,
    # 'Calibration':Cali_inferencer,
    'ICL_PPL': ICL_PPL_inferencer
}

def build_inferencer(inferencer_type, **kwargs):
    return inferencer_dict[inferencer_type](**kwargs)