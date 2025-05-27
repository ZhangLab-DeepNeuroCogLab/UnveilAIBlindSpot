# Unveiling AI’s Blind Spots: An Oracle for In-Domain, Out-of-Domain, and Adversarial Errors

Authors: Shuangpeng Han, Mengmi Zhang

This is a PyTorch implementation of our paper. **Our paper has been accepted in ICML 2025.**

<p align="left">
  <a href="https://arxiv.org/pdf/2410.02384v3" target="_blank"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  &nbsp
  <a href="" target="_blank"><img src="https://img.shields.io/badge/Presentation-Video-blue.svg"></a>
  &nbsp
  <a href="https://drive.google.com/file/d/13eNm1NPxhRnXnWWln6Os0B1CvtcEzPEp/view?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Paper-Poster-green.svg"></a>
</p> 

## Project Description

AI models make mistakes when recognizing images—whether in-domain, out-of-domain, or adversarial. Predicting these errors is critical for improving system reliability, reducing costly mistakes, and enabling proactive corrections in real-world applications such as healthcare, finance, and autonomous systems. However, understanding what mistakes AI models make, why they occur, and how to predict them remains an open challenge. Here, we conduct comprehensive empirical evaluations using a 'mentor' model—a deep neural network designed to predict another 'mentee' model’s errors. Our findings show that the mentor excels at learning from a mentee's mistakes on adversarial images with small perturbations and generalizes effectively to predict in-domain and out-of-domain errors of the mentee. Additionally, transformer-based mentor models excel at predicting errors across various mentee architectures. Subsequently, we draw insights from these observations and develop an 'oracle' mentor model, dubbed SuperMentor, that can outperform baseline mentors in predicting errors across different error types from the ImageNet-1K dataset. Our framework paves the way for future research on anticipating and correcting AI model behaviors, ultimately increasing trust in AI systems.

<br>
<p align="center"><img align="center"  src="./Images/intro.jpg" alt="..." width="1920">
</p>

Our experiments show that mentors trained on adversarial images of a mentee outperform mentors trained on out-of-domain (OOD) and in-domain (ID) images of the same mentee on (a) CIFAR-10, (b) CIFAR-100 and (c) ImageNet-1K datasets.
<br>
<p align="center"><img align="center"  src="./Images/error_type.jpg" alt="..." width="1920">
</p>

## Environment Setup

Our code is based on Pytorch 1.12.1 and Python 3.9.19.

We recommend using [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) for installation:


```
conda env create -f environment.yml

conda activate oracle

```

## Dataset

You can download the in-domain (ID) and out-of-domain (OOD) image datasets described in our paper from the link provided:

* [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet-1K](https://www.image-net.org/download.php)
* [CIFAR-10-C/CIFAR-100-C/ImageNet-1K-C](https://www.cs.toronto.edu/~kriz/cifar.html)

Please use the adversarial attack implementations from [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) to generate the adversarial (AA) images, applying the following four attack methods:

* PGD 
* CW 
* Jitter 
* PIFGSM 

## Training & Testing

Our codebase is structured in a manner similar to the [SlowFast](https://github.com/facebookresearch/SlowFast/tree/main) repository.

All hyperparameters are listed and explained in the [Config](Config/defaults.py).

Remember to modify the following parameters in the [YAML](YAML/BMP.yaml) file to your own:
* BASE_MODEL.CHECKPOINT_FILE_PATH
* TRAIN.DATASET
* TEST.DATASET
* OUTPUT_DIR

You can start training the model from scratch by running:

```
python3 Main/main.py --cfg YAML/train.yaml
```
We use TRAIN.ENABLE and TEST.ENABLE in the [YAML](YAML/BMP.yaml) file to determine if training or testing should be performed for the current task. If you wish to conduct only testing, set TRAIN.ENABLE to False. 

You can test the model by running:

```
python3 Main/main.py --cfg YAML/train.yaml \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False 
```

## Citation

If you find our work useful in your research, please use the following BibTeX entry for citation.

```
@article{han2024unveiling,
  title={Unveiling AI's Blind Spots: An Oracle for In-Domain, Out-of-Domain, and Adversarial Errors},
  author={Han, Shuangpeng and Zhang, Mengmi},
  journal={arXiv preprint arXiv:2410.02384},
  year={2024}
}
```



