from .vqa import Answer_Extractor
from .utils import Base_Metric
class Answer_Extractor_map(Answer_Extractor): # TODO 
    def __init__(self, choices='ABCDEFG') -> None:
        self.choices = choices
    # Prefetch Answers
    def infer_option(self, answer,item_choices):
        def get_unit_option(splits, choices='ABCD', prefix='', suffix=''):
            res = None
            for c in choices:
                if prefix + c + suffix in splits:
                    if res is None:
                        res = c
                    else:
                        return None
            return res
        splits = [x.strip() for x in answer.split()]

        # no prefix match
        no_prefix_option = get_unit_option(splits, item_choices)
        if no_prefix_option is not None and no_prefix_option != 'A':
            return no_prefix_option

        # prefix match
        tups = [('(', ')'), ('(', ').'), ('', '.'), ('', ','), ('', ':'), ('', ')'), ('', ').'), 
                (':', ''), (':', ','), (':', '.'), (':', ')'), (':', ').')]
        for tup in tups:
            prefix_option = get_unit_option(splits, item_choices, prefix=tup[0], suffix=tup[1])
            if prefix_option is not None:
                return prefix_option
        return None

    def infer_text(self, answer, choices,item_choices):
        answer = answer.lower()
        assert isinstance(choices, list)
        gt_choices = {}
        for idx, k in enumerate(choices):
            gt_choices[item_choices[idx]] = str(k).lower()
        cands = []
        for key, value in gt_choices.items():
            if value in answer:
                cands.append(key)
        if len(cands) == 1:
            return cands[0]
        return None

    def preprocess_text(self, answer):
        output_text = answer
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        # mmbench direct pattern
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            return '(' + res[0][:-1] + ')'
        # ppl pattern
        pattern = re.compile(r'\([A-Z]')
        res = pattern.findall(output_text)
        if len(res) > 0:
            return res[0] + ')'
        return answer

    def fetch_answer(self, answer, choices, item_choices):
        answer = self.preprocess_text(answer)
        copt = self.infer_option(answer,item_choices)
        #return copt if copt else self.infer_text(answer, choices,item_choices)
        return copt
    
class MMBench_Calibration(Base_Metric):
    def __init__(self, dataset_name, content_only = False):
        super().__init__(dataset_name)
        self.choices = 'ABCD'
        self.match_option = 0
        self.match_content = 0
        self.answer_extractor = Answer_Extractor(content_only)

    def eval_sub_data(self, sub_data, answer_map):
        lt = len(sub_data)
        GT, PRED = [], []
        result = 1
        for i in range(lt):
            item = sub_data[i]
            idx = item['id']
            GT.append(self.choices[answer_map[idx]])
            pred_answer, option_match, content_match = self.answer_extractor.fetch_answer(item['answer'], item['gt_choices'])
            PRED.append(pred_answer)
            if pred_answer is not None:
                self.match_content += content_match
                self.match_option += option_match
                if GT[-1] != PRED[-1]:
                    result = 0
            else:
                result = 0
        return result

    def metric_func(self, answers):
        vanilla_score, circular_score = 0.0, 0.0
        vanilla_cnt = 0
        result = {}
        answer_map = {} # gt
        cnt = len(answers)
        for item in answers:
            answer_map[item['id']] = item['gt_choice']
        answers = sorted(answers, key = lambda i: int(i['id']))
        answers_unique = []
        for i in tqdm(range(len(answers)), desc="Running Metric"):
            idx = answers[i]['id']
            main_idx = str(int(idx) % int(1e6))
            if main_idx in result:
                continue

            ## vanilla
            vanilla_cnt += 1
            answers[i]['correct']=0
            pred_option, _, _ = self.answer_extractor.fetch_answer(answers[i]['answer'], answers[i]['gt_choices'])
            if pred_option == self.choices[answer_map[answers[i]['id']]]:
                vanilla_score += 1
                answers[i]['correct']=1
            
            sub_data = []
            for j in range(len(answers)):
                if int(answers[j]['id']) % int(1e6) == int(idx):
                    sub_data.append(answers[j])
            out = self.eval_sub_data(sub_data, answer_map)
            circular_score += out
            result[main_idx] = out
            answers_unique.append(answers[i])
        #import ipdb;ipdb.set_trace()
        num_bins=10
        probs = []
        accs = []
        least = int(len(answers_unique) / num_bins)
        plus_max_id = len(answers_unique) % num_bins -1
        cur_bin_id=0
        cur_bin_max = least+1 if plus_max_id >=0 else least
        cali=0.0
        cur_bin = []
        cur_bin_acc = [] 
        sorted_answers = sorted(answers_unique, key=lambda x: x['prob'])
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
        assert total == len(answers_unique)
        cali = cali / len(answers_unique) 
        return dict(
            vanilla_acc = vanilla_score / vanilla_cnt * 100,
            circular_acc = circular_score / vanilla_cnt * 100,
            option_match = self.match_option / cnt * 100,
            content_match = self.match_content /cnt *100,
            ece=cali,
            acc_bins=accs,
            probs=probs
        )


class ScienceQA_Calibration(Base_Metric):
    CHOICE = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J']
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
    def check_text(self, text, choices, gt_id):
        text = text.lower()
        if choices[gt_id].lower() not in text:
            return False
        for id, choice in enumerate(choices):
            if id == gt_id:
                continue
            if choice.lower() in text:
                return False
        return True
    def check_option(self, res_list, gt_char):
        for res in res_list:
            if gt_char not in res:
                return False
        return True
    def check_pattern2(self, res_list, gt_char):
        pred = res_list[0][-1]
        if pred == gt_char:
            return True
        return False
    def metric_func(self, answers):
        import re
        pattern_1 = re.compile(r'The answer is \(?[A-E]\W|the answer is \(?[A-E]\W')
        pattern_2 = re.compile(r'ANSWER: [A-E]')
        pattern_3 = re.compile(r'\([A-E]')
        
        score = 0.0
        for item in tqdm(answers, desc="Running Metric"):
            tmp_score = 0
            gt_choice = item['gt_choice']
            gt_char = self.CHOICE[gt_choice]
            pred_text = item['answer']
            pred_text = pred_text
            res_1 = pattern_1.findall(pred_text)
            res_2 = pattern_2.findall(pred_text)
            res_3 = pattern_3.findall(pred_text)
            if len(res_1) !=0 :
                if self.check_option(res_1, gt_char):
                    tmp_score = 1.0
            elif len(res_2) !=0:
                if self.check_pattern2(res_2, gt_char):
                    tmp_score = 1.0
            elif len(res_3) != 0:
                if self.check_option(res_3, gt_char):
                    tmp_score = 1.0
            elif self.check_text(pred_text, item['gt_choices'], gt_choice):
                tmp_score = 1.0
            score+=tmp_score
            
            item['correct']=1 if tmp_score>0 else 0
        
        score = score/len(answers) * 100
        num_bins=10
        probs = []
        accs = []
        least = int(len(answers) / num_bins)
        plus_max_id = len(answers) % num_bins -1
        cur_bin_id=0
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
        return {'acc':score, 'ece': cali,'acc_bins':accs, 'probs':probs}




class LAMM_VQA_INF(Base_Metric):
    CHOICE = 'ABCDEFG'
    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name)
        self.answer_extractor = Answer_Extractor_map()

    def metric_func(self, answers):
        score = 0.0
        result = {}
        vanilla_cnt = 0
        for item in tqdm(answers, desc="Running Metric"):
            idx = item['id']
            main_idx = str(int(idx) % int(1e6))
            if main_idx in result:
                continue
            vanilla_cnt += 1
            gt_choice = item['gt_choice']
            gt_char = item['options'][gt_choice]
            pred_text = item['answer']
            pred_option = self.answer_extractor.fetch_answer(pred_text, item['gt_choices'],item['options'])
            #import ipdb;ipdb.set_trace()
            if pred_option!=None and pred_option.lower() == gt_char.lower():
                score += 1.0
            result[main_idx]=1
        score = score/vanilla_cnt * 100
        return score


class POPE_Metric:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    
    def metric_func(self, answers):
        label_list=[]
        for answer in answers:
            text = answer['answer']
            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]
            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answer['answer'] = 'no'
            else:
                answer['answer'] = 'yes'
            if answer['gt_answers'] =='no':
                label_list.append(0)
            else:
                label_list.append(1)

        pred_list = []
        for answer in answers:
            if answer['answer'] == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)
        
        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)
        
        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1
        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))
        results={
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1,
            'Yes ratio': yes_ratio
        }
        return results

    def metric(self, answer_path):
        with open(answer_path, 'rb') as f:
            answers = json.load(f)
        results = self.metric_func(answers) 
        print(f'{self.dataset_name}:')
        for key, value in results.items():
            print(f'{key}: {value}')
        return results