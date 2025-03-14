## Introduction
The code repository is for the paper "Efficient Feature Fusion for UAV Object Detection".

Citations:
@article{wang2025efficient,
  title={Efficient Feature Fusion for UAV Object Detection},
  author={Wang, Xudong and Shen, Chaomin and Peng, Yaxin},
  journal={arXiv preprint arXiv:2501.17983},
  year={2025}
}

The features of the paper is added on top of YOLO-v10.  Please refer to https://github.com/THU-MIG/yolov10.

Model:
![image](https://github.com/user-attachments/assets/8b47d704-fe1e-4d84-8489-24ca7912d2c8)

FMSA:
![image](https://github.com/user-attachments/assets/97dec717-971f-424f-b49f-4f9bdebb175c)

FDS:
![image](https://github.com/user-attachments/assets/23a5bc67-5661-47ac-991c-934a39b9b6bf)

FUS:
![image](https://github.com/user-attachments/assets/453dfab3-f14f-47f2-9289-f3f2a0bed643)


## Installation
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
pip install einops
```

## Training
python tony_train.py


## Code
Please refer to yolov10/ultralytics/nn/modules/cv_attention.py for the core functionalities of the paper.



