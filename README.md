DL project

# Deformable-DETR

This an implementation of Deformable-DETR. 
reference Codes are based on [DETR](https://github.com/facebookresearch/detr) project.
[Deformable attention](https://github.com/Windaway/Deformable-Attention-for-Deformable-DETR/blob/main/DFMAtt.py)

# Preparation

For DETR stuffs, etc. data preparation, evaluation, and others , please refer to 
[DETR](https://github.com/facebookresearch/detr).

# Training

## The training is on the machine with 8 v100 GPU,  it will takes 6 days training 300 epochs from scratch

Below is the training script for DDP training.
```shell script
bash train.sh
```

For single gpu training, try the code below

```python
python main.py
--output_dir my_output \
--coco_path ~/dev/data/coco \
--lr 0.0002 \
--lr_backbone 0.00001 \
--num_queries 300 \
--batch_size 1 \
--enc_layers 6 \
--dec_layers 6 \
--no_aux_loss \
--amp
```
or
```python
sh train.sh
```

If you do not need AMP to accelerate training , just remove this flag.

# extend of DETR

- 1.backbone
  - change lr for projection layers
  - add backbone modifications for returning multi-scale feature maps

- 2.attention
  - add Multi-scale Deformabe Attention Module
  - integrate MS-Deformable-Attention into DETR architecture
  - modify transfomer's implementation to be adapted to Deformable-Attention
  - add image mask to MS-Deformable-Attention

- 3.loss
  - add focal loss for classification

- 4.optimizer
  - add adam for the optimizer

- 5.speed up
  - add AMP
  - add Prefetch DataLoader

- 6.notebook
  - add finetuning notkbook
  - make finetuing dataset
