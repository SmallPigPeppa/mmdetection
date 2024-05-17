CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py
WORK_DIR=work_dirs/vitdet_256x256_vanilla
python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work-dir ${WORK_DIR}