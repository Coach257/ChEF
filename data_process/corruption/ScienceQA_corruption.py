
import os
from PIL import Image
import numpy as np

import random
import json
from tqdm import tqdm
import torchvision.transforms as trn
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Corruption")
    parser.add_argument("--corrupt_type", default='text', choices=['text','image'])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    return args

def fix_seed(seed_value=2024):  
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def img_corrpution(args):
    from image_corruption_method import d_noise, d_blur, d_weather, d_digital, other

    data_path = args.data_path
    image_dir = os.path.join(data_path, 'sqaimage_images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]

    image_data = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)  
        img_array = np.array(img)
        image_data[image_file] = img_array

    d_crp=[d_noise,d_blur,d_weather,d_digital]

    fix_seed()
    Image.MAX_IMAGE_PIXELS = None

    
    
    with open('ScienceQA_imgcrp_order.json','rb') as f:
        orders = json.load(f)
    save_path = os.path.join(args.save_path, 'ScienceQA')
    img_save_path = os.path.join(save_path,'sqaimage_images')
    os.makedirs(img_save_path, exist_ok=True)

    crp_info = []

    for id in tqdm(orders):
        im = image_data[id]
        tmp=[]
        for crp in d_crp:
            random_key = random.choice(list(crp.keys()))
            severity = random.randint(1,5)
            corruption = lambda clean_img: crp[random_key](clean_img, severity)
            tmp.append({
            'corruption_type': random_key,
            'severity':severity
            })
            #print(random_key)
            convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
            final_img = np.uint8(corruption(convert_img(im)))
            im=final_img

        for k in other.keys():
            # 可视化变换前的图像
            severity = random.randint(1,5)
            corruption = lambda clean_img: other[k](clean_img, severity)
            tmp.append({
            'corruption_type': k,
            'severity':severity
            })
            convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
            #img = pad_image_to_multiple(img, multiple=32)
            #print(img.shape)
            final_img = np.uint8(corruption(convert_img(im)))
            im=final_img

        crp_info.append({
            'id':id,
            'crp_seq':tmp
        })
        pil_image = Image.fromarray(final_img)
        output_path = os.path.join(img_save_path, id)
        #print(output_path)
        pil_image.save(output_path)


    with open(os.path.join(save_path,'Image_Corruptions_info.json'),'w') as f:
        json.dump(crp_info,f,indent=4)
    #print(crp_info)


def text_corruption(args):
    from text_corruption import d_basic,d_sentence, d_word, d_character, d_option
    data_path = args.data_path
    json_path = os.path.join(data_path, 'meta_file','VQA_ScienceQA.json')
    with open(json_path, 'rb') as f:
        data = json.load(f)
    
    fix_seed()
    crp_info = []
    crp_res = []
    levels = ['sentence','word','character']
    d_crp = [d_sentence, d_word, d_character]

    save_path = os.path.join(args.save_path, 'ScienceQA')
    os.makedirs(save_path, exist_ok=True)
    
    for sample in tqdm(data):
        text = sample['query']
        qlist = text.split('Options:')
        q = qlist[0].split('Context:')
        context = q[1]
        text = q[0]
        #basic
        sample_info = {'id':sample['id'],
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
                        print(f'Error on {sample["id"]} when use {random_key} method, retrying...')
                        continue
                    ptext = text
                    break
                    
                except:
                    print(f'Error on {sample["id"]} when use {random_key} method, retrying...')
                    continue
        #option

        c_or_r = random.choice(['circular_option','reverse_option'])
        text, gt_choices, gt_choice = d_option[c_or_r](text, sample['gt_choices'], sample['gt_choice'], context, opt_pr='Options:')
        sample_info['crps'].append({"level": "option",
                            "method": c_or_r,
                            "severity": 1})
        crp_sample = sample.copy()
        crp_sample['query'] = text
        crp_sample['gt_choices'] = gt_choices
        crp_sample['gt_choice'] = gt_choice
        if crp_sample['gt_choices'][crp_sample['gt_choice']] != sample['gt_choices'][sample['gt_choice']]:
            import ipdb;ipdb.set_trace()
        assert crp_sample['gt_choices'][crp_sample['gt_choice']] == sample['gt_choices'][sample['gt_choice']]
        crp_res.append(crp_sample)
        crp_info.append(sample_info)


    with open(os.path.join(save_path,'VQA_ScienceQA_C.json'),'w') as f:
        json.dump(crp_res,f,indent=4)
    with open(os.path.join(save_path,'Text_Corruptions_info.json'),'w') as f:
        json.dump(crp_info,f,indent=4)
    



def main():
    args = parse_args()
    if args.corrupt_type == 'image':
        img_corrpution(args)
    elif args.corrupt_type == 'text':
        text_corruption(args)
    



if __name__ == '__main__':
    main()