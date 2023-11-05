
import os
from PIL import Image
import numpy as np
from image_corruption_method import d_noise, d_blur, d_weather, d_digital, other
import random
import json
from tqdm import tqdm
import torchvision.transforms as trn
import argparse
import pandas as pd
import base64
import io
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Corruption")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--corrupt_type", default='text', choices=['text','image'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.corrupt_type == 'text':
        text_corruption(args)
    elif args.corrupt_type == 'image':
        image_corruption(args)

def fix_seed(seed_value=2024):  
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def image_corruption(args):
    data_path = args.data_path
    df = pd.read_csv(os.path.join(data_path, 'mmbench_dev_20230712.tsv'), sep='\t')
    

    def decode_base64_to_image(base64_string):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    
    image_arrays = []

    for i in range(len(df)):
        image = df.iloc[i]['image']
        image = decode_base64_to_image(image)
        image_arrays.append(np.array(image))


    d_crp=[d_noise,d_blur,d_weather,d_digital]


    fix_seed()
    Image.MAX_IMAGE_PIXELS = None
    crp_info = []
    save_path = os.path.join(args.save_path, 'MMBench')
    img_save_path = os.path.join(save_path, 'images')
    os.makedirs(img_save_path, exist_ok=True)
    for id, im in tqdm(enumerate(image_arrays)):
        tmp=[]
        for crp in d_crp:
            while True:
                try:
                    random_key = random.choice(list(crp.keys()))
                    severity = random.randint(1,5)
                    corruption = lambda clean_img: crp[random_key](clean_img, severity)
                    tmp.append({
                    'corruption_type': random_key,
                    'severity':severity
                    })
                    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
                    final_img = np.uint8(corruption(convert_img(im)))
                    im=final_img
                    break
                except:
                    print(f'Error on {id} when use {random_key} method, retrying...')
                    continue

                
        for k in other.keys():
            severity = random.randint(1,5)
            corruption = lambda clean_img: other[k](clean_img, severity)
            tmp.append({
            'corruption_type': k,
            'severity':severity
            })
            convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
            final_img = np.uint8(corruption(convert_img(im)))
            im=final_img

        crp_info.append({
            'id':id,
            'crp_seq':tmp
        })
        pil_image = Image.fromarray(final_img)
        output_path = os.path.join(img_save_path, str(id)+'.png')
        #print(output_path)
        pil_image.save(output_path)
        

    with open(os.path.join(save_path,'Image_Corruptions_info.json'),'w') as f:
        json.dump(crp_info,f,indent=4)

def load_from_df(df, idx, key):
        if key in df.iloc[idx] and not pd.isna(df.iloc[idx][key]):
            return df.iloc[idx][key]
        else:
            return None
        

def text_corruption(args):
    from text_corruption import d_basic,d_sentence, d_word, d_character, d_option

    data_path = args.data_path
    df = pd.read_csv(os.path.join(data_path, 'mmbench_dev_20230712.tsv'), sep='\t')
    sys_prompts = 'There are several options:'
    option_candidate = ['A', 'B', 'C', 'D', 'E']
    fix_seed()
    crp_info = []
    crp_res = []
    levels = ['sentence','word','character']
    d_crp = [d_sentence, d_word, d_character]

    save_path = os.path.join(args.save_path, 'MMBench')
    os.makedirs(save_path, exist_ok=True)


    for idx in tqdm(range(len(df))):
        question = df.iloc[idx]['question']
        hint = load_from_df(df, idx, 'hint')
        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: load_from_df(df, idx, cand) 
            for cand in option_candidate
            if load_from_df(df,idx, cand) is not None
        }
        gt_choices = [value for value in options.values()]
        answer = df.iloc[idx]['answer'] if 'answer' in df.iloc[0].keys() else None
        gt_choice = option_candidate.index(answer) if answer is not None else None

        text = f'{hint} {question}'
        index = int(df.iloc[idx]['index'])
        #basic
        sample_info = {'id': index, 
                       'crps':[{
                            "level": "basic",
                            "method": "lowercase",
                            "severity": 1
                        }]
                        }
        text = d_basic['lowercase'](text)
        c_or_e = random.choice(['constraction','expansion'])
        text = d_basic[c_or_e](text)
        sample_info['crps'].append({"level": "basic",
                            "method": c_or_e,
                            "severity": 1})
        #'sentence','word','character'
        #import ipdb;ipdb.set_trace()
        ptext = text
        for level, crp in zip(levels, d_crp):
            while True:
                try:
                    random_key = random.choice(list(crp.keys()))
                    severity = 1
                    if 'char_' in random_key or 'swap_syn' in random_key:
                        severity = random.randint(1,5)
                        text = crp[random_key](text, severity=severity)
                    else:
                        text = crp[random_key](text)
                    sample_info['crps'].append({"level": level,
                                    "method": random_key,
                                    "severity": severity})
                    if text == None or len(text)<=1:
                        text = ptext
                        print(f'Error on {index} when use {random_key} method, retrying...')
                        continue
                    ptext = text
                    break
                    
                except:
                    print(f'Error on {index} when use {random_key} method, retrying...')
                    continue
        #option

        c_or_r = random.choice(['circular_option','reverse_option'])
        text, new_choices, new_choice = d_option[c_or_r](text, gt_choices, gt_choice, opt_pr='There are several options:')
        sample_info['crps'].append({"level": "option",
                            "method": c_or_r,
                            "severity": 1})
        crp_sample = {}
        crp_sample['id'] = int(index)
        crp_sample['query'] = text
        crp_sample['gt_choices'] = new_choices
        crp_sample['gt_choice'] = new_choice
        if crp_sample['gt_choices'][crp_sample['gt_choice']] != gt_choices[gt_choice]:
            import ipdb;ipdb.set_trace()
        assert crp_sample['gt_choices'][crp_sample['gt_choice']] == gt_choices[gt_choice]
        crp_res.append(crp_sample)
        crp_info.append(sample_info)
        #import ipdb;ipdb.set_trace()
        

    with open(os.path.join(save_path,'MMBench_C.json'),'w') as f:
        json.dump(crp_res,f,indent=4)
    with open(os.path.join(save_path,'Text_Corruptions_info.json'),'w') as f:
        json.dump(crp_info,f,indent=4)
    



if __name__ == '__main__':
    main()