# joint_chestxray

Joint learning of chest radiographs and radiology reports in the application of pulmonary edema assessment.

This repository incorporates the algorithms presented in <br />
G. Chauhan<sup>\*</sup>, R. Liao<sup>\*</sup> et al. Joint Modeling of Chest Radiographs and Radiology Reports for Pulmonary Edema Assessment. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2020. <br />
(<sup>\*</sup> indicates equal contribution)

# Instructions

## Setup

## Training

## Testing

# Notes on Data and Labels

## MIMIC-CXR

We have experimented this algorithm on [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), which is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA.

## Pulmonary edema severity

We have demonstrated the application of this algorithm in pulmonary edema assessment. We aim to clssify a given chest x-ray image into one of the four ordinal levels: no edema (0), vascular congestion (1), interstitial edema (2), and alveolar edema (3).

## Regex and expert labeling

We use [regex](https://github.com/RayRuizhiLiao/regex_pulmonary_edema) to extract pulmonary edema severity labels from radiology reports for the model training. A board-certified radiologist and two domain experts reviewed 485 radiology reports (corrsponsding to 531 chest radiographs). We use the expert labels for our model testing.

# Contact

Geeticka Chauhan: geeticka [at] mit.edu <br />
Ruizhi Liao: ruizhi [at] mit.edu
