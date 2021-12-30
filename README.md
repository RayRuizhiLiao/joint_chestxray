# joint_chestxray

Joint learning of chest radiographs and radiology reports in the application of pulmonary edema assessment.

This repository incorporates the algorithms presented in <br />
G. Chauhan<sup>\*</sup>, R. Liao<sup>\*</sup> et al. [Joint Modeling of Chest Radiographs and Radiology Reports for Pulmonary Edema Assessment.](https://arxiv.org/pdf/2008.09884.pdf) *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2020. <br />
(<sup>\*</sup> indicates equal contributions)

# Instructions

## Setup

Set up the conda environment using [`conda_environment.yml`](https://github.com/RayRuizhiLiao/joint_chestxray/blob/master/conda_environment.yml). You might want to remove the pip dependencies if that is creating an issue for you. You can manually install the spacy and scispacy dependencies using `pip install spacy` and `pip install scispacy`. Read [`https://allenai.github.io/scispacy/`](https://allenai.github.io/scispacy/) for more information on scispacy. 

## BERT

Download the pre-trained BERT model, tokenizer, etc. from [`Dropbox`](https://www.dropbox.com/sh/hl00bp2rtrykbvy/AACiMaEzKS95hpwv9WvaMTg6a?dl=0). You should download the folder *scibert_scivocab_uncased* that contains five files. The path to *scibert_scivocab_uncased* should be passed to [`--bert_pretrained_dir`](https://github.com/RayRuizhiLiao/joint_chestxray/blob/42d5c8bb10adc9edbbb696b0d27b6b735403b339/scripts/parser.py#L23).
     
## Training

Train the model in an unsupervised fashion, i.e., only the first term in [Eq (3)](https://arxiv.org/pdf/2008.09884.pdf) is optimized:

```
python ${repo_path}/scripts/main.py
--img_data_dir ${repo_path}/example_data/images/
--text_data_dir ${repo_path}/example_data/text/
--data_split_path ${repo_path}/example_data/data_split.csv
--use_text_data_dir
--use_data_split_path
--output_dir ${output_path}
--do_train
--training_folds 1 2 3 4 5 6
--training_mode 'semisupervised_phase1'
```

Train the model in a supervised fashion:

```
python ${repo_path}/scripts/main.py
--img_data_dir ${repo_path}/example_data/images/
--text_data_dir ${repo_path}/example_data/text/
--data_split_path ${repo_path}/example_data/data_split.csv
--use_text_data_dir
--use_data_split_path
--output_dir ${output_path}
--do_train
--training_folds 1 2 3 4 5
--training_mode 'supervised'
```
Note that in our [data split](https://github.com/RayRuizhiLiao/joint_chestxray/blob/master/example_data/data_split.csv), fold 6 is the unlabeled image-text pairs. 

# Notes on Data and Labels

## MIMIC-CXR

We have experimented this algorithm on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), which is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA.

## Pulmonary edema severity

We have demonstrated the application of this algorithm in pulmonary edema assessment. We aim to classify a given chest x-ray image into one of the four ordinal levels: no edema (0), vascular congestion (1), interstitial edema (2), and alveolar edema (3).

## Radiology report pre-processing

We extract the *Impressions*, *Findings*, *Conclusion*, and *Recommendation* sections from the radiology reports. If none of these sections are present in the report, we use the *Final Report* section. We tokenize the text using [ScispaCy](https://allenai.github.io/scispacy/) before providing it to the BERT model. The data pre-processing code is available [here](https://github.com/RayRuizhiLiao/joint_chestxray/tree/master/joint_img_txt/data_preprocessing). We're releasing the extracted report sections soon! 

## Regex and expert labeling

We use [regex](https://github.com/RayRuizhiLiao/regex_pulmonary_edema) to extract pulmonary edema severity labels from the radiology reports for our model training. A board-certified radiologist and two domain experts reviewed and labeled 485 radiology reports (corrsponsding to 531 chest radiographs). We use the expert labels for our model testing. The regex labeling results and expert labels on MIMIC-CXR are summerized [here](https://github.com/RayRuizhiLiao/joint_chestxray/blob/master/metadata/mimic-cxr-sub-img-edema-split-manualtest.csv).

## Data split

In our MICCAI 2020 work, we split the MIMIC-CXR data into training and test sets. There is no patient overlap between the training set and the test set. Our data split can be found [here](https://github.com/RayRuizhiLiao/joint_chestxray/blob/master/metadata/mimic-cxr-sub-img-edema-split-allCXR.csv). The folds 1-6 are the training set and the fold "TEST" is the test set. We also used the training set for cross-validation when tuning our model hyper-parameters.

# Contact

Geeticka Chauhan: geeticka [at] mit.edu <br />
Ruizhi Liao: ruizhi [at] mit.edu
