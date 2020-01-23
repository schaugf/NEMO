# Neural Estimator of Metastatic Origin

Geoffrey F. Schau, Erik A. Burlingame, Guillaume Thibault, Tauangtham Anekpuritanang, Ying Wang, Joe W. Gray, Christopher Corless, Young Hwan Chang, "Predicting Primary Site of Secondary Liver Cancer with a Neural Estimator of Metastatic Origin (NEMO)"

You can read the initial preprint on bioRxiv [here](https://www.biorxiv.org/content/10.1101/689828v1)

<img src='assets/nemo_concept.png'>

## Overview

Pathological inspection of stained biopsy tissue remains a gold-standard process for the diagnosis of various types of cancer and other malignancies. 
In the case of metastatic cancer that has spread from beyond the primary site and to distal organ systems, histopathological assessment of biopsies can provide essential information to best guide diagnosis and treatment options for the patient.

This study sought to evaluate whether deep learning systems are capable of distinguishing the metastatic origin of whole-slide histopathological images taken from biopsies of liver metastases.
We present a two-step approach that first trains a model to identify regions of tissues that contain cancerous mass and filter out normal liver, necrosis, and stromal tissue from the training set.
After the learned first model is applied to the entire dataset, we then train a second-stage model to correctly classify regions of tissue according to their metastatic origin which in our dataset originates from one of three classes: gastrointestinal stroma, neuroendocrine carcinomas, or adenocarcinomas. 
Finally, we compare our model's performance to that of three board-certified pathologists and illustrate that our model achieves competitive performance in this task.

In this study, we utilize the [inception v4](https://github.com/kentsommer/keras-inceptionV4/blob/master/inception_v4.py) learning architecture.


## Stage 1

The first stage model is designed to learn to correctly classify regions of whole slide images as either tumor tissue or normal liver or stroma. 

### Pathological Annotations

Pathological annotations of tumor/non-tumor regions were provided by an expert pathologist through a secure instance of an OMERO server containing relevant images.
All annotations are downloaded and stored as binary mask files for downstream data parsing, and are accessed with a simply utility script.

```bash
python src/get_annotations.py
```

### Preprocessing

Whole slide images are first loaded and tiled into non-overlapping patches of 299x299 pixels.
White background tiles are removed with a mean intensity threshold cutoff, and remaining tiles are normalized with an established method of histopathological color intensity normalization, included [here](https://github.com/schaugf/HEnorm_python).
Slides can be preprocessed individually or in parallel on a slurm-equipped computing cluster.

```bash
python stage1_preprocess.py --slide_file
sbatch run/stage1_preprocess.submit
```

### Data Parsing



### Training


### Evaluation




## Stage 2

### Preprocessing

With Stage 1

### Training

### Evaluation

## Pathologist Comparison



