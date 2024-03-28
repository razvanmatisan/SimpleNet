# SimpleNet


![](imgs/cover.png)

**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**

[Paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)

##  Introduction

This repository contains code for **SimpleNet** implemented in PyTorch that can also use a CLIP backbone (i.e., ViT-B/32) as a feature extractor.

## SimpleNet with CLIP Backbone - implementation details

The code for the CLIP Backbone can be found in `common.py`, where the class `CLIPFeatureExtractor` contains the logic of the feature extractor component. In short, it aims to return a list of features that were obtained after the input was passed through the layers mentioned in the command line. For example, if we set the parameters `-le` to 2 and 5, then the CLIP Feature Extractor will return a list of 2 feature vectors resulted from the layer 2 and 5, respectively.

## Get Started 

### Environment 
The setup can be made by running the following commands in a terminal:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

Edit `run.sh` to edit dataset class and dataset path.

#### MvTecAD

Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

The dataset folders/files follow its original structure.

### Run

#### Demo train

Please specicy dataset path (line1) and log folder (line10) in `run.sh` before running.

`run.sh` gives the configuration to train models on MVTecAD dataset.
```
bash run.sh
```
#### Training using CLIP backbone
In order to run the code using a CLIP backbone, you can change the bash script `run.sh` by using the following parameters:
```
-b clip \
-le 2 \
-le 3 \
--pretrain_embed_dimension 768 \
--target_embed_dimension 768 \
--patchsize 10 \
--meta_epochs 40 \
```

It is important to mention that the indices of the layers (i.e., '-le' parameters) need to be integers from 0 to 11 (because ViT-B/32 has 12 attention blocks).

One could perform hyperparameter tuning to find the best values of the parameters. However, the values mentioned above lead to competitive results, comparable with SimpleNet with wideresnet50.


## Citation
```
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20402--20411},
  year={2023}
}
```

## Acknowledgement

Thanks for great inspiration from [PatchCore](https://github.com/amazon-science/patchcore-inspection)

## License

All code within the repo is under [MIT license](https://mit-license.org/)
