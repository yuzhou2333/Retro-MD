#!/usr/bin/env bash

# 控制是否启用wandb
USE_WANDB=false  # 设置为false可以禁用wandb

# wandb相关配置
if [ "$USE_WANDB" = true ]; then
    export WANDB_API_KEY=""  # 替换为您的wandb API密钥
    export WANDB_NAME=""  # 实验名称
    export WANDB_TAGS=""  # 可选：添加标签
    WANDB_ARGS="--use-wandb --wandb-project your-project-name --wandb-entity your-entity-name" # 对your-project-name与your-entity-name修改为您的实体与名称
else
    WANDB_ARGS=""
fi

python_path=/root/miniconda3/bin/python
project_path="/root/autodl-tmp/Retro-MD"
reaction_pairs="src1-tgt1,src2-tgt2,src3-tgt3,src4-tgt4,src5-tgt5,src6-tgt6,src7-tgt7,src8-tgt8,src9-tgt9,src10-tgt10"
path_2_data=/root/autodl-tmp/Retro-MD/data_bin/uspto50k_aug20
reaction_list=${path_2_data}/reaction_list.txt

SAVE_DIR=${project_path}/save_models/Retro_MD2
mkdir -vp $SAVE_DIR

CUDA_VISIBLE_DEVICES=0
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_retro --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1 \
  --sampling-temperature-2 10 \
  --mutual-distillation-mode "automatic" \
  --step-size-scheduler "inverse_sqrt_root2" \
  --trial-dataset-ratio 0.2 \
  --weight-update-interval 1 \
  --decoder-langtok \
  --encoder-langtok "tgt" \
  --reaction-dict "$reaction_list" \
  --reaction-pairs "$reaction_pairs" \
  --criterion knowledge_distillation_criterion --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.001 \
  --share-decoder-input-output-embed \
  --max-epoch 30 \
  --dropout 0.3 --weight-decay 0.01 \
  --max-tokens 16384 --update-freq 1 \
  --save-interval 1 \
  --seed 222 --log-format simple --log-interval 10 \
  --bpe sentencepiece \
  --pure-batch \
  --RS-epoch \
  $WANDB_ARGS > ${SAVE_DIR}/train.log 2>&1


bash ${project_path}/PCL_scripts/uspto50k_aug20/test.sh ${SAVE_DIR}/model1
bash ${project_path}/PCL_scripts/uspto50k_aug20/test.sh ${SAVE_DIR}/model2
