model_cfg=configs/models/llava.yaml
recipe_cfg=configs/desiderata_recipes/Insfollow/ScienceQA.yaml
YOUR_PARTITION=<YOUR_PARTITION>
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${YOUR_PARTITION} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python tools/eval_insfollow.py --model_cfg=${model_cfg} --recipe_cfg=${recipe_cfg} --debug
