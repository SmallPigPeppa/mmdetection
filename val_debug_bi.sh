CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294_256x256_bi.pth
CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e-256x256-pinew.py
WORK_DIR=work_dirs/vitdet_256x256_bi
python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work-dir ${WORK_DIR}