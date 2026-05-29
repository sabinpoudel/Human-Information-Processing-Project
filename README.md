This repository contains the assignment work for the Human Information Processing project. The project focuses on analyzing physiological signals, 
especially ECG data, and extracting heart rate variability features using the WESAD dataset. The main goal is to understand how physiological signals can be 
used to study human states such as stress, relaxation, and emotional response.The work is implemented in a Jupyter Notebook and includes data processing, signal analysis,
feature extraction, and visualization. Repository Contents
Human-Information-Processing-Project/
│
├── HIP.ipynb          # Main Jupyter Notebook for the assignment
├── README.md         # Project description and instructions figures are included, the pipelines model design of the neuropype
This project is based on the WESAD dataset, which contains wearable sensor data collected for stress and affect detection. 
The dataset includes physiological signals such as ECG, EDA, respiration, temperature, and accelerometer data.
For this assignment, the focus is mainly on ECG-based analysis and HRV-related features.
The link to dataset is :https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection #(Due to large size of the dataset and limitation of the 
github the dataset is not directly uploaded here. However, it can be easily obtained via link 

# How to Run the Project

Download or clone this repository:
git clone https://github.com/sabinpoudel/Human-Information-Processing-Project.git
Open the project folder:
cd Human-Information-Processing-Project
Launch Jupyter Notebook:
jupyter notebook
Open the notebook:
HIP.ipynb
Run the notebook cells from top to bottom.
Requirements

Install the required Python libraries before running the notebook:





# ECG-Derived Heart Rate Variability Analysis for Stress Detection Using NeuroPype
#### Sabin, Student ID: 113034860, Course : Human Information Processing 


This project investigates whether ECG-derived heart rate variability can distinguish baseline and stress states during human information processing. Stressful cognitive and emotional conditions activate the autonomic nervous system, especially sympathetic activity, while often reducing parasympathetic or vagal regulation. These autonomic changes are reflected in the timing between heartbeats. Therefore, ECG-derived heart rate variability provides a useful physiological index for studying autonomic reactions to stress.

This project built a complete ECG: HRV stress-analysis workflow in two environments. Firtly, WESAD chest ECG was extracted and converted to per-condition to CSV files in this jupyter notebook. As well as filtered, visually checked and R-peaks were detected. Furthermore, HRV windows were computed, group statistics were run, and a simple JAX classifier. Correspondigly,the working NeuroPype implementation mirrors that logic in two forms: an offline importCSV pipeline for reproducible batch processing and export. The key practical lesson is that NeuroPype ImportCSV produces an offline single-packet recording due to which TimeSeriesPlot is often blank unless the data is streamed; the docs state TimeSeriesPlot is intended for continuous streaming data, while ScrollPlot /offline viewers are for prerecorded review. SpectrumPlot also requires a frequency-axis spectrum, so it must be preceded by a spectral estimation node such as Spectrogram or WelchSpectrum . HeartRate and HeartRateVariability must consume the RDetection output stream ending in -peaks. 


The project uses the WESAD dataset, a public wearable physiology dataset designed for stress and affect detection. The main signal used in this project is chest ECG recorded from the RespiBAN device at 700 Hz. The analysis compares baseline and stress segments. The core hypothesis is that stress will increase heart rate and reduce HRV compared with baseline. Specifically, stress is expected to reduce time-domain HRV metrics such as SDNN, RMSSD, SDSD, NN50, and pNN50.

The project includes both offline and pipeline-based analysis. In Jupyter Notebook(herein), the ECG is extracted, filtered, visually inspected, artifact-marked, and converted into HRV features. Figures are generated, including raw and filtered ECG time-series plots, PSD before/after filtering, artifact-marked ECG, R-peak visualization, paired baseline-vs-stress HRV plots, and latent-space visualizations using PCA, t-SNE, and UMAP. A simple JAX logistic-regression model is also used to test whether extracted HRV features contain stress-relevant information.

Finally, the preprocessed ECG is streamed into NeuroPype using Lab Streaming Layer. The NeuroPype pipeline demonstrates signal input, filtering, R-peak detection, heart-rate extraction, HRV feature extraction, and final computational output. 


## Workflow

1. Load WESAD `.pkl` files.
2. Extract chest ECG.
3. Keep only baseline and stress labels.
4. Convert data to CSV.
5. Filter ECG.
6. Visualize raw and filtered ECG.
7. Generate PSD before/after filtering.
8. Mark artifacts.
9. Detect R-peaks.
10. Extract HRV features.
11. Compare baseline vs stress.
13. Visualize HRV feature space using PCA, t-SNE, and UMAP.
14. Export clean ECG for NeuroPype.
15. The detail elaboration of the figure and result discussion


# Tools Used
Python
Jupyter Notebook
NumPy
Pandas
Matplotlib
SciPy
NeuroPype / signal processing tools
GitHub

# Project Objectives

The main objectives of this assignment are:

1) To load and explore physiological signal data.
2) To analyze ECG signals from the WESAD dataset.
3) To process raw ECG data for further analysis.
4) To extract HRV-related information.
5) To visualize signal patterns and results.
6) To understand the relationship between physiological signals and human information processing.

# Discussion of the results obtained in details:

### note: Please download HIP ipynb file to visualize the figure
## HRV Paired Plots

The HRV paired plots compare each subject’s baseline and stress values for the selected HRV metrics. Each thin line represents one subject and shows the direction of change from baseline to stress.

This paired visualization is useful because it shows within-subject physiological changes rather than only comparing group averages. The thick black line represents the group mean trend and summarizes the overall direction of change across subjects.

---

## Mean Heart Rate Plot

The mean heart rate plot shows a clear increase from baseline to stress for most subjects.

This indicates that heart rate rises during the stress condition. Physiologically, this pattern is consistent with sympathetic nervous system activation during stress. When stress increases, cardiovascular activity also increases, leading to faster heartbeats.

**Conclusion:** Stress increases heart rate.

---

## Mean NN Interval Plot

The mean NN interval plot shows that most subjects have lower NN intervals during stress compared with baseline.

NN interval represents the time between consecutive normal heartbeats. A decrease in NN interval means that the time between heartbeats becomes shorter. This is consistent with the observed increase in heart rate.

**Conclusion:** Stress shortens the time between heartbeats.

---

## SDNN Plot

The SDNN plot shows mixed subject-level changes between baseline and stress.

Some subjects show a decrease, while others show little change or an increase. SDNN reflects overall variability in normal-to-normal heartbeat intervals. Because the direction of change is not consistent across subjects, this plot does not show a clear stress-related pattern.

**Conclusion:** SDNN does not show a consistent stress effect in this notebook.

---

## RMSSD Plot

The RMSSD plot shows variable responses across subjects.

Some subjects show lower RMSSD during stress, while others remain similar or increase. RMSSD reflects short-term heart rate variability and is commonly associated with parasympathetic regulation.

Although stress may reduce parasympathetic activity, the pattern here is not consistent enough across subjects to support a reliable decrease.

**Conclusion:** RMSSD does not show a statistically reliable stress-related decrease in this analysis.

---

## pNN50 Plot

The pNN50 plot shows that many subjects have lower pNN50 values during stress compared with baseline, although not all subjects follow this pattern.

pNN50 measures the percentage of adjacent NN intervals differing by more than 50 ms. Lower pNN50 values suggest reduced beat-to-beat variability, which may reflect reduced parasympathetic activity during stress.

However, because the decrease is not consistent across all subjects and the Wilcoxon p-value is borderline, this should be interpreted as a possible reduction rather than a strong confirmed effect.

**Conclusion:** pNN50 shows a likely stress-related reduction, but the evidence is borderline.

---

## Binary Cross-Entropy Loss Plot

The binary cross-entropy loss plot shows model loss across training epochs.

The x-axis represents epoch, and the y-axis represents loss. The curve decreases smoothly during training, indicating that the logistic-regression model is learning from the HRV features.

A smooth decrease in loss suggests that optimization is numerically stable. There is no clear evidence of unstable training, divergence, or strong oscillation.

**Conclusion:** The logistic-regression model is learning, and training is numerically stable.

---

## PCA Latent-Space Plot

The PCA latent-space plot projects the HRV features into two principal components.

The x-axis represents the first principal component, and the y-axis represents the second principal component. The colors represent baseline and stress conditions.

The first principal component explains 75.7% of the variance, while the second principal component explains 15.5%. Together, these two components capture most of the variation in the HRV feature set.

The plot shows partial separation between baseline and stress, suggesting that HRV features contain condition-related structure. However, overlap remains between the two groups, meaning that baseline and stress are not perfectly separable using these features alone.

**Conclusion:** HRV features contain stress-related structure, but baseline and stress still overlap.

---

## t-SNE Latent-Space Plot

The t-SNE latent-space plot shows a nonlinear embedding of the HRV windows.

The axes are arbitrary t-SNE dimensions and do not have direct physiological meaning. Points that are close together have similar HRV feature profiles. Colors indicate baseline and stress conditions.

Some local regions appear enriched for one condition, suggesting that certain HRV patterns are more common during either baseline or stress.

However, t-SNE should not be overinterpreted because it can create apparent clusters depending on parameter settings and local neighborhood structure.

**Conclusion:** Some local clusters are condition-enriched, but the plot should be interpreted cautiously.

---

## UMAP Latent-Space Plot

The UMAP latent-space plot shows another nonlinear embedding of the HRV feature space.

The x-axis represents UMAP dimension 1, and the y-axis represents UMAP dimension 2. These axes are embedding coordinates and should not be interpreted as direct physiological measurements.

The plot shows some separation between baseline and stress points, suggesting that HRV features contain information related to stress condition. However, overlap remains between the two groups.

This overlap indicates that HRV features do not perfectly distinguish stress from baseline, likely because of individual differences and physiological variability.

**Conclusion:** Some condition separation exists, but HRV features do not perfectly separate stress from baseline.

---

## JAX Latent Stress Score Plot

The JAX latent stress score plot shows predicted stress probability for baseline and stress windows.

The x-axis represents condition, and the y-axis represents predicted stress probability. The boxplot summarizes the distribution of predicted probabilities, while the jittered points show individual windows.

Stress windows generally receive higher predicted stress probabilities than baseline windows. This suggests that the model learned a stress-related signal from the HRV features.

However, there is overlap between the baseline and stress distributions. Some baseline windows receive relatively high stress probabilities, and some stress windows receive lower probabilities.

**Conclusion:** Stress windows generally receive higher stress probabilities, but the classifier is not perfect.

---

## Overall Summary

The plots provide visual support for the stress hypothesis. The strongest and most consistent effects are observed in mean heart rate and mean NN interval. Heart rate increases during stress, while NN interval decreases, showing faster cardiac activity under stress.

Some HRV variability measures, such as pNN50, suggest a possible reduction during stress, but the evidence is weaker and less consistent. SDNN and RMSSD show mixed subject-level responses and do not demonstrate a clear stress effect.

The PCA, t-SNE, UMAP, and JAX stress score plots show that HRV features contain condition-related information. However, overlap between baseline and stress remains, indicating that HRV-based stress detection is informative but not perfectly separable.







Depending on the notebook code, additional libraries may be required.
