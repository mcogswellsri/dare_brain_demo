#!/bin/bash


###################################################
# training
#python train.py --model unet data/models/unet/

#python train.py --model unet --train-epochs 600 data/models/unet_longer/

#python train.py --model unet --train-epochs 1000 data/models/unet_longer_1k/

#python train.py --model local_unet --train-epochs 600 data/models/local_unet/

#python train.py --model local_unet --train-epochs 600 data/models/local_unet_side30-50/

#python train.py --model unet --train-epochs 1000 --up-mode resize data/models/unet_resize2/




###################################################
# debug training
#python train.py --model local_unet --max-examples 2 --train-epochs 10 data/models/local_unet_debug/

#python train.py --model unet --up-mode resize --max-examples 10 --train-epochs 10 data/models/unet_resize_debug

#python train.py --model unet --max-examples 3 --train-epochs 10 data/models/unet_up4_debug





###################################################
# debug evaluation


## generate submission
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 0 --contest-submission 1 --save-metrics 0 \
#    --val-data /dataSRI/DataSets/BrATS/MICCAI_BraTS2020_ValidationData/ \
#    --result-dname first_submission \
#    --checkpoint-epoch 1000 data/models/unet_resize2/

## generate nifti vis files
#VK='entire_cls0'
#LK='model.0'
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 1 --contest-submission 0 \
#    --vis-key $VK --layer-key $LK \
#    --result-dname an_vis_niftis_${VK}_${LK} \
#    --checkpoint-epoch 1000 data/models/unet_resize2/ \
#    #--max-examples 4
#python analyze.py \
#    --vis-key $VK --layer-key $LK \
#    --result-dname an_vis_niftis_${VK}_${LK} \
#    data/models/unet_resize2/

## evaluate model and save metrics
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 0 --contest-submission 0 --save-metrics 1 \
#    --result-dname main_evaluation \
#    --checkpoint-epoch 1000 data/models/unet_resize2/
#python eval.py --model unet \
#    --vis-page 0 --vis-nifti 0 --contest-submission 0 --save-metrics 1 \
#    --result-dname main_evaluation \
#    --checkpoint-epoch 1000 data/models/unet_longer_1k/

## compare visualizations
#mkdir -p data/models/unet_resize2/full_vis_comparison_with_rankcorr_try2/
#python analyze.py --result-dname full_vis_comparison_with_rankcorr_try2 data/models/unet_resize2/

## generate vis files for isaac
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 1 --contest-submission 0 \
#    --result-dname first_four_all_vis_niftis \
#    --checkpoint-epoch 1000 data/models/unet_resize2/ \
#    --max-examples 4

## generate webpage of examples
#python eval.py --model unet --up-mode resize \
#    --vis-page 1 --vis-nifti 0 --contest-submission 0 \
#    --result-dname vis_examples_uniform_scale \
#    --max-examples 40 --checkpoint-epoch 1000 data/models/unet_resize2/
#python vis_utils.py --result-dname vis_examples_uniform_scale data/models/unet_resize2/





###################################################
# generate webpage of examples from scratch




# generate webpage of examples
python eval.py --model unet --up-mode resize \
    --vis-page 1 --vis-nifti 0 --contest-submission 0 \
    --result-dname examples_v2_uncertain_pt1 \
    --pred-thres 0.5 \
    --max-examples 3 --checkpoint-epoch 1000 data/models/unet_resize2/
python vis_utils.py --result-dname examples_v2_uncertain_pt1 data/models/unet_resize2/


## generate alpha data for analysis
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 0 --contest-submission 0 --alpha-analysis 1 \
#    --result-dname alpha_analysis \
#    --pred-thres 0.5 \
#    --max-examples 100 --checkpoint-epoch 1000 data/models/unet_resize2/



## generate features for survival prediction
#python eval.py --model unet --up-mode resize \
#    --vis-page 0 --vis-nifti 0 --contest-submission 0 --survival-features 1 \
#    --split all \
#    --result-dname survival_features \
#    --pred-thres 0.005 \
#    --checkpoint-epoch 1000 data/models/unet_resize2/







# test different thresholds
#declare -a thresholds=(0.01 0.03 0.1 0.2 0.3 0.5)
#
#for thresh in "${thresholds[@]}"
#do
#    python eval.py --model unet --up-mode resize \
#        --vis-page 0 --vis-nifti 0 --save-metrics 1 --contest-submission 0 --alpha-analysis 0 \
#        --result-dname "performance_${thresh}" \
#        --pred-thres $thresh \
#        --max-examples 100 --checkpoint-epoch 1000 data/models/unet_resize2/
#done
