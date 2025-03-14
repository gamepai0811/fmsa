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



