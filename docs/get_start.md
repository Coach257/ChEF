# Getting Started

## Prepare Environment
```shell
conda create -n ChEF python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements/run.txt
```

## Data Preparation

### LAMM
Download LAMM 2D Benchmark datasets. More details are in [LAMM](https://github.com/OpenLAMM/LAMM). 

- CIFAR10
- Flickr30k
- FSC147
- ScienceQA
- VOC2012

The datasets should have this structure:

```text
ChEF
├── configs
└── data
    └── LAMM
        └── 2D_Benchmark
            ├── cifar10_images
            ├── flickr30k_images
            ├── fsc147_images
            ├── meta_file
            ├── sqaimage_images
            └── voc2012_images
```

### Omnibenchmark
Download [Omnibenchmark](https://entuedu-my.sharepoint.com/:f:/g/personal/yuanhan002_e_ntu_edu_sg/El2wmbzutJBOlu8Tz9HyDJABMmFtsG_8mq7uGh4Q7F1QSQ?e=NyroDS) for fine-grained classification dataset and [Bamboo Label System](https://github.com/ZhangYuanhan-AI/Bamboo) for hierarchical catergory labels. 

The dataset should have this structure:

```text
ChEF
├── configs
└── data
    ├── Omnibenchmark_raw
    └── Bamboo
        └── sensexo_visual_add_academic_add_state_V4.visual.json
```

We sampled and labeled Omnibenchmark meticulously by using
a hierarchical chain of categories, facilitated by the Bamboo label system. 
```shell
python data_process/Omnibenchmark.py
```

You can also directly download the labeled Omnibenchmark dataset from [OpenXLab](https://openxlab.org.cn/datasets?lang=zh-CN). # TODO

Finally, the dataset should have this structure:

```text
ChEF
├── configs
└── data
    ├── ChEF
    |   └── Omnibenchmark_Bamboo
    |       ├── meta_file
    |       └── omnibenchmark_images
    └── Bamboo
        └── sensexo_visual_add_academic_add_state_V4.visual.json
```

### MMBench, MME and SEEDBench
Refer to [MMBench](https://github.com/open-compass/MMBench), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) and [SEEDBench](https://github.com/AILab-CVC/SEED-Bench) for dataset and more details.

The datasets should have this structure:

```text
ChEF
├── configs
└── data
    ├── MMBench
    |   ├── mmbench_dev_20230712.tsv
    |   └── mmbench_test_20230712.tsv
    ├── MME_Benchmark_release_version
    └── SEED-Bench
```


### POPE
POPE is a special labeled COCO dataset for hallucination evaluation based on the validation set of COCO 2014. Download [COCO](https://cocodataset.org/#download)  and [POPE](https://github.com/RUCAIBox/POPE).

The datasets should have this structure:

```text
ChEF
├── configs
└── data
    └── coco_pope
        ├── val2014
        ├── coco_pope_adversarial.json
        ├── coco_pope_popular.json
        └── coco_pope_random.json
```

### MMBench_C and ScienceQA_C
MMBench_C and ScienceQA_C are datasets with image and text corruptions fot robustness evaluation. 
We recommend that you directly download the MMBench_C and ScienceQA_C dataset from [OpenXLab](https://openxlab.org.cn/datasets?lang=zh-CN). # TODO

If you want to generate MMBench_C and ScienceQA_C from the original dataset, run the following data processing commands after downloading [MMBench](#mmbench-mme-and-seedbench) and ScienceQA from [LAMM](#lamm). 



```shell
cd data_process/corruption
conda create -n corruption python=3.10.0
conda activate corruption
pip install -r requirements.txt

#for MMBench text/image corruption
python MMBench_corruption.py  \
 --data_path /path/to/MMBench  \
 --save_path /path/to/save  \
 --corrupt_type text  
 # --corrupt_type image

#for ScienceQA text/image corruption
python ScienceQA_corruption.py  \
 --data_path /path/to/LAMM/2D_Benchmark  \
 --save_path /path/to/save  \
 --corrupt_type text  
 # --corrupt_type image

```
Although we fixed random seeds, there are still random factors in the generation process (e.g. results may vary when stylizing text with styleformers), so we still recommend that you download MMBench_C and ScienceQA_C directly rather than generate from scratch.


Finally, the dataset should have this structure:

```text
ChEF
├── configs
└── data
    └── ChEF
        ├── MMBench_C
        |   ├── images
        |   ├── Image_Corruptions_info.json
        |   ├── Text_Corruptions_info.json
        |   └── MMBench_C.json
        └── ScienceQA_C
            ├── sqaimage_images
            ├── Image_Corruptions_info.json
            ├── Text_Corruptions_info.json
            └── VQA_ScienceQA_C.json
```
## MLLMs Preparation

See [models.md](models.md) for details. 

## Evaluation

We provide several recipes and model configs in ChEF/configs. 
For example, to evaluate [LAMM](https://github.com/OpenLAMM/LAMM) on CIFAR10 using the default recipe, 
run:
```shell
python eval.py \
    --model_cfg configs/models/lamm.yaml \
    --recipe_cfg configs/scenario_recipes/CIFAR10/default.yaml
```

To evaluate the desiderata, see [desiderata.md](desiderata.md) for details.

To deploy your own model or design your own recipe, see [tutorial.md](tutorial.md) for details.