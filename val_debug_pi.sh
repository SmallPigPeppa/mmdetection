#CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
#CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py
#WORK_DIR=work_dirs/vitdet_1024x1024
#python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work_dir ${WORK_DIR}



#CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth
#CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e-512x512.py
#WORK_DIR=work_dirs/vitdet_512x512
#python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work-dir ${WORK_DIR}


CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294_512x512_pi.pth
CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e-512x512-pi.py
WORK_DIR=work_dirs/vitdet_512x512_pi
python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work-dir ${WORK_DIR}


CHECKPOINT_PATH=ckpt/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294_256x256_pi.pth
CONFIG_PATH=projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e-256x256-pi.py
WORK_DIR=work_dirs/vitdet_256x256_pi
python tools/test.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --work-dir ${WORK_DIR}
