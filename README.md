# Segment Anything

## Train
### standalone
```shell
python train.py -c configs/flare_box_finetune.yaml 
```
### distributed
```shell
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py -o distributed=True -c configs/flare_box_finetune.yaml 
```

## Inference on One Image
```shell
python inference.py
```

## Evaluation
### standalone
```shell
python eval.py -c configs/flare_box_finetune.yaml 
```
### distributed
```shell
torchrun --standalone --nnodes=1 --nproc_per_node=8 eval.py -o distributed=True -c configs/flare_box_finetune.yaml 
```
