# ReFAA
Official PyTorch implementation for the paper - **Residual Feature Alignment and Aggregation for Multimodal Disinformation Detection**.

## Environment
```
python==3.10
torch==2.0.1
(pip install -r requirements.txt)
```
Please also install groudingdino in [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository.

## Dataset

Please refer to [DGM4](https://github.com/rshaojimmy/MultiModal-DeepFake), [MSD_Data](https://github.com/Lbotirx/CofiPara/tree/master/data/mmsd2/README.md) and [MSTI_Data](https://github.com/Lbotirx/CofiPara/tree/master/data/msti/README.md)

## Training and Test
Edit model_config.py to change training and test settings.
```
python main.py
```

## Acknowledgements
This project is built on the open source repository [CofiPara](https://github.com/viczxchen/CofiPara). Thanks the team for their impressive work!