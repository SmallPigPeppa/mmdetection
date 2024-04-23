CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
python tools/test.py projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py ${CHECKPOINT_PATH}