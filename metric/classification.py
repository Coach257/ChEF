import os
import json
from tqdm import tqdm
from .utils import Base_Metric

class CG_Classification(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
        from .utils import classification_acc, Cleaner
        self.cleaner = Cleaner()
        self.syn_check_func = classification_acc
    
    def metric_func(self, answers):
        correct = 0
        syn_correct = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            pred = self.cleaner.clean(pred)
            if self.syn_check_func(gt, pred):
                syn_correct += 1
            if gt in pred.split():
                correct += 1
        acc = correct / len(answers) * 100
        syn_acc = syn_correct / len(answers) * 100
        return dict(
            ACC = acc,
            SYN_ACC = syn_acc,
        )

class FG_Classification(Base_Metric):
    
    def __init__(self, dataset_name, bamboo_tree_path = None, inference_type = 'direct', **kwargs):
        super().__init__(dataset_name)         
        self.bamboo_tree_path = bamboo_tree_path
        annot_data = json.load(open(self.bamboo_tree_path,'rb'))
        self.id2name = annot_data['id2name']
        self.father2child = annot_data['father2child']
        name2id = {}
        for key, value in self.id2name.items():
            for name in value:
                name2id[name] = key
        self.name2id = name2id
        self.child2father = annot_data['child2father']
        self.inference_type = inference_type
        assert self.inference_type in ['direct', 'single_ppl', 'multi_ppl']

    def weighted_ACC_multi(self, pred, gt):
        assert len(pred) == len(gt)
        for i in range(len(pred)):
            if pred[i] != gt[i]:
                return i/len(pred)
        return 1

    def weighted_ACC_single(self, pred, gt):
        deep_idx = 0
        for i in range(len(gt)):
            if pred == gt[i]:
                deep_idx = i+1
        return deep_idx/len(gt)
    
    def share_father(self, father_id, pred):
        children = self.father2child[father_id]
        for child in children:
            if child in pred:
                return True
        return False

    def weighted_ACC_direct(self, pred, gt):
        deep_idx = 0
        for i in range(len(gt)):
            if gt[i] in pred:
                deep_idx = i+1
            else:
                father = self.name2id[gt[i-1]] if i>0 else self.child2father[self.name2id[gt[i-1]]][0]
                if self.share_father(father, pred):
                    deep_idx = i
        return deep_idx/len(gt)

    def metric_func(self, answers):
        wcorrect = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            if self.inference_type == 'multi_ppl':
                wcorrect += self.weighted_ACC_multi(pred, gt)
            elif self.inference_type == 'single_ppl':
                wcorrect += self.weighted_ACC_single(pred, gt)
            else:
                wcorrect += self.weighted_ACC_direct(pred, gt)

        wacc = wcorrect / len(answers) * 100
        return wacc

class LAMM_Classification_WC(Base_Metric):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
        from .tools import classification_acc, VQAEval
        self.cleaner = VQAEval()
        self.syn_check_func = classification_acc
    def metric_func(self, answers):
        correct = 0
        for item in tqdm(answers, desc="Running Metric"):
            gt = item['gt_answers']
            pred = item['answer']
            pred = self.cleaner.clean(pred)
            item['correct']=0
            if gt in pred.split():
                correct += 1
                item['correct']=1
        num_bins=10
        probs = []
        accs = []
        least = int(len(answers) / num_bins)
        plus_max_id = len(answers) % num_bins -1
        cur_bin_id=0
        cur_bin_len=0
        cur_bin_max = least+1 if plus_max_id >=0 else least
        cali=0.0
        cur_bin = []
        cur_bin_acc = [] 
        sorted_answers = sorted(answers, key=lambda x: x['prob'])
        total = 0
        #import ipdb;ipdb.set_trace()
        for item in tqdm(sorted_answers, desc="Running Calibration Metric"):
            cur_bin.append(item['prob'])
            cur_bin_acc.append(item['correct'])
            if len(cur_bin) == cur_bin_max:
                avg_p = sum(cur_bin)*1.0 / len(cur_bin)
                probs.append(avg_p)
                avg_a = sum(cur_bin_acc)*1.0 / len(cur_bin_acc)
                accs.append(avg_a)
                cali+= len(cur_bin)*abs(avg_p-avg_a)
                total += len(cur_bin_acc)
                cur_bin_id+=1
                cur_bin = []
                cur_bin_acc = []
                if cur_bin_id==plus_max_id+1:
                    cur_bin_max-=1
        #import ipdb;ipdb.set_trace()
        assert total == len(answers)
        cali = cali / len(answers) 
        acc = correct / len(answers) * 100
        return {'acc':acc, 'ece': cali,'acc_bins':accs, 'probs':probs}