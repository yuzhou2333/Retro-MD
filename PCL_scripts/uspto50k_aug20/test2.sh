#!/usr/bin/env bash

python_path=/root/miniconda3/bin/python
project_path="/root/autodl-tmp/Retro-MD"
reaction_pairs="src1-tgt1,src2-tgt2,src3-tgt3,src4-tgt4,src5-tgt5,src6-tgt6,src7-tgt7,src8-tgt8,src9-tgt9,src10-tgt10"
path_2_data=/root/autodl-tmp/Retro-MD/data_bin/uspto50k_aug20
reaction_list=${path_2_data}/reaction_list.txt

checkpoint_path="/root/autodl-tmp/Retro-MD/save_models/Retro_MD/model2"
checkpoint_name=checkpoint30.pt
translate_path="/root/autodl-tmp/Retro-MD/results/model2/30"
if [ -n "$2" ]; then
  checkpoint_name=$2
fi
model=${checkpoint_path}/${checkpoint_name}
echo "model: ${model}"
OUTPUT_DIR=$checkpoint_path
TRANSLATION_DIR=$translate_path
mkdir -p $TRANSLATION_DIR

# CUDA_VISIBLE_DEVICES=0
for i in {1..10}; do
    src="src$i"
    tgt="tgt$i"
    ${python_path} ${project_path}/fairseq_cli/generate.py $path_2_data \
        --path $model \
        --task translation_multi_simple_epoch \
        --reaction-dict "$reaction_list" \
        --reaction-pairs "$reaction_pairs" \
        --gen-subset test \
        --source-reaction $src \
        --target-reaction $tgt \
        --encoder-langtok "tgt" \
        --scoring sacrebleu \
        --remove-bpe 'sentencepiece' \
        --batch-size 128 \
        --decoder-langtok > $TRANSLATION_DIR/test_${src}_${tgt}.txt 2>&1
done

${python_path} ${project_path}/PCL_scripts/uspto50k_aug20/result_statistics.py $TRANSLATION_DIR > ${TRANSLATION_DIR}/result.txt 2>&1

