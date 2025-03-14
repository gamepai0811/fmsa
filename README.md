## Introduction
The code repository is for the paper "Efficient Feature Fusion for UAV Object Detection".

Paper: https://arxiv.org/abs/2501.17983

![image](https://github.com/user-attachments/assets/c5dbc594-3147-4292-94b0-bd861695d620)

![image](https://github.com/user-attachments/assets/d8905927-dd8b-4024-afb1-0039db9a659a)

![image](https://github.com/user-attachments/assets/676425b0-7883-4260-9db3-a863dff21ad2)



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


## Our features
Please refer to yolov10/ultralytics/nn/modules/cv_attention.py for the core functionalities of the paper.


## Acknowledgement
The code base is built with YOLO-v10.  Please refer to https://github.com/THU-MIG/yolov10.

Thanks for the great implementations!


## Citation
If our code or models help your work, please cite our paper:
```
@article{wang2025efficient,
  title={Efficient Feature Fusion for UAV Object Detection},
  author={Wang, Xudong and Shen, Chaomin and Peng, Yaxin},
  journal={arXiv preprint arXiv:2501.17983},
  year={2025}
}
```

