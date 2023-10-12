from .Direct import Direct_inferencer
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import copy_batch_dict
import numpy as np

class PPL_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
        for batch in tqdm(dataloader, desc="Running inference"):
            if self.CoT:
                prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
            else:
                prompts = self.instruction_handler.generate_basic_query(batch)
                cot = None
            
            batch_options = batch['options']
            image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, _ = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, CoT = cot)
            outputs = model.ppl_inference(image_path, questions, answers, answer_options, CoT_answer)

            ppl_np = np.array(outputs)
            for idx in range(len(batch['image_path'])):
                ppl_results = ppl_np[ppl_batch_mask[idx]]
                pred_answer_id = ppl_results.argmin()
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = questions[ppl_batch_mask[idx].argmax()]
                answer_dict['ppl_results'] = ppl_results.tolist()
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[idx]
                answer_dict['answer'] = batch['options'][idx][pred_answer_id]
                predictions.append(answer_dict)

        self._after_inference_step(predictions)


class ICL_PPL_inferencer(Direct_inferencer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def inference(self, model, dataset):
        predictions=[]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            if self.CoT:
                prompts, cot = self.instruction_handler.generate_CoT_query(model, batch)
            else:
                prompts = self.instruction_handler.generate_basic_query(batch)
                cot = None
            ices = self.instruction_handler.generate_ices(prompts, batch_idx, self.batch_size)
            
            batch_options = batch['options']
            image_path, questions, answers, ppl_batch_mask, answer_options, CoT_answer, ices = self.instruction_handler.generate_ppl_query(prompts, batch, batch_options, ices, CoT = cot)
            outputs, icl_prompts = model.icl_ppl_inference(image_path, questions, answers, answer_options, ices, self.instruction_handler.icl_cfg, CoT_answer)
            ppl_np = np.array(outputs)
            icl_prompt_idx = 0
            for idx in range(len(batch['id'])):
                ppl_results = ppl_np[ppl_batch_mask[idx]]
                pred_answer_id = ppl_results.argmin()
                answer_dict = copy_batch_dict(batch, idx)
                answer_dict['query'] = icl_prompts[icl_prompt_idx]
                answer_dict['ppl_results'] = ppl_results.tolist()
                if self.CoT:
                    answer_dict['CoT_answer'] = cot[idx]
                answer_dict['answer'] = batch['options'][idx][pred_answer_id]
                predictions.append(answer_dict)
                icl_prompt_idx += len(ppl_results)

        self._after_inference_step(predictions)