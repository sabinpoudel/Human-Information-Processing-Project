This repository contains the assignment work for the Human Information Processing project. The project focuses on analyzing physiological signals, 
especially ECG data, and extracting heart rate variability features using the WESAD dataset. The main goal is to understand how physiological signals can be 
used to study human states such as stress, relaxation, and emotional response.The work is implemented in a Jupyter Notebook and includes data processing, signal analysis,
feature extraction, and visualization. 
Repository Contents

→ HIP.ipynb   💡 Main Jupyter Notebook for the assignment

→ README.md      💡 Project description and instructions figures are included, the pipelines model design of the neuropype

---

This project is based on the WESAD dataset, which contains wearable sensor data collected for stress and affect detection. 
The dataset includes physiological signals such as ECG, EDA, respiration, temperature, and accelerometer data.
For this assignment, the focus is mainly on ECG-based analysis and HRV-related features.

The link to dataset is :https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection (Where Dataset is described)
Direct : https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx?openfile=true (To directly download the Folder)

💡 (Due to large size of the dataset and limitation of the  github the dataset is not directly uploaded here. However, it can be easily obtained via link provided herein.)

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



## Introduction

This project investigates whether ECG-derived heart rate variability (HRV) can differentiate between baseline and stress states during human information processing. When individuals are exposed to cognitively demanding or emotionally stressful conditions, the autonomic nervous system adjusts rapidly to support attention, decision-making, and coping demands. This adjustment is typically marked by increased sympathetic activation and reduced parasympathetic, or vagal, regulation.

These changes do not appear only as a general increase in arousal; they are also expressed in the fine timing patterns between consecutive heartbeats. Under baseline conditions, heart rhythms usually show greater variability, reflecting flexible parasympathetic regulation and adaptive physiological balance. During stress, however, this variability often becomes more constrained, suggesting a shift toward sympathetic dominance and reduced autonomic flexibility.

By analyzing ECG-derived HRV, this project aims to infer how the body’s regulatory systems respond to different information-processing demands. In this sense, HRV serves not merely as a cardiac measurement, but as a physiological window into the interaction between cognitive load, emotional stress, and autonomic control

This project developed a complete ECG–HRV stress-analysis workflow across two complementary environments: a Jupyter-based analytical pipeline and a NeuroPype-based signal-processing implementation. In the Jupyter notebook, chest ECG data from the WESAD dataset were extracted and reorganized into condition-specific CSV files(Please check HIP.ipynb file for it), allowing baseline and stress recordings to be processed separately. The ECG signals were then filtered, visually inspected for quality, and analyzed through R-peak detection. From these detected heartbeat intervals, HRV features were computed over defined time windows, followed by group-level statistical analysis and the implementation of a simple JAX-based classifier.

This workflow supports the inference that stress-related changes in autonomic regulation can be observed not only in raw ECG morphology but also in the variability patterns derived from consecutive R-peaks. By comparing HRV windows across experimental conditions, the pipeline provides a structured way to examine whether baseline and stress states produce separable physiological signatures.

The NeuroPype implementation was designed to mirror the same logic in a more modular signal-processing environment. An important implementation insight concerns frequency-domain visualization. SpectrumPlot does not operate directly on raw ECG or HRV time-series data; it requires frequency-axis spectral input. Therefore, it must be preceded by a spectral estimation node such as Spectrogram or WelchSpectrum. Similarly, HeartRate and HeartRateVariability nodes must receive the output stream from RDetection, specifically the stream ending in -peaks, because HR and HRV calculations depend on detected R-peak timing rather than the raw ECG waveform itself.

Overall, the project demonstrates both the analytical and practical requirements of ECG-based HRV stress analysis. The Jupyter workflow provides transparent feature extraction, statistical comparison, and classification, while the NeuroPype workflow translates the same reasoning into a reusable signal-processing pipeline. Together, these implementations show how ECG-derived R-peak timing can be transformed into interpretable HRV measures for distinguishing physiological responses during baseline and stress-related information processing.

The project uses the WESAD dataset, a publicly available wearable physiology dataset developed for stress and affect detection. The primary physiological signal analyzed in this work is chest ECG recorded from the RespiBAN device at 700 Hz, which provides high-resolution cardiac timing information suitable for heart rate and HRV analysis. The study focuses on comparing baseline and stress segments in order to examine whether autonomic changes during stress can be detected through ECG-derived HRV.

The central hypothesis is that stress will produce a measurable shift in cardiac regulation compared with baseline. Under stress, increased sympathetic activation is expected to elevate heart rate, while reduced parasympathetic or vagal activity is expected to decrease HRV. Therefore, stress is expected to reduce time-domain HRV metrics such as SDNN, RMSSD, SDSD, NN50, and pNN50. These reductions would suggest that the heartbeat intervals become less variable and less flexible during stress, indicating a move toward sympathetic dominance and reduced autonomic adaptability.

Methodologically, the project combines offline analysis with a pipeline-based implementation. In the Jupyter Notebook, chest ECG is extracted from WESAD, separated into baseline and stress conditions, filtered to reduce noise, visually inspected for signal quality, and marked for artifacts where necessary. R-peaks are then detected from the cleaned ECG signal, allowing inter-beat intervals to be calculated and transformed into HRV features. This step is important because the project does not treat ECG only as a waveform; instead, it uses ECG as a source for inferring autonomic regulation through heartbeat timing.

Several visual and analytical outputs are generated to support interpretation. These include raw and filtered ECG time-series plots, power spectral density plots before and after filtering, artifact-marked ECG visualizations, R-peak detection plots, paired baseline-versus-stress HRV comparisons, and latent-space visualizations using PCA, t-SNE, and UMAP. Together, these figures make the analysis more illustrative by showing how the signal changes across preprocessing stages and whether baseline and stress samples form separable patterns in feature space.

A simple JAX logistic-regression classifier is also implemented to test whether the extracted HRV features contain stress-relevant information. Rather than serving as a complex predictive model, this classifier functions as an inferential check: if the model can distinguish baseline from stress above chance level, it suggests that the HRV features preserve physiologically meaningful differences between conditions.

Finally, the project extends the analysis into NeuroPype by streaming the preprocessed ECG through Lab Streaming Layer. The NeuroPype pipeline demonstrates the practical translation of the offline workflow into a modular signal-processing environment. It includes signal input, filtering, R-peak detection, heart-rate extraction, HRV feature extraction, and final computational output. This pipeline-based implementation shows that the same ECG-to-HRV reasoning can be reproduced beyond the notebook, supporting both offline analysis and future real-time physiological monitoring applications.


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

## Raw ECG Visual Inspection Plot

<img width="3600" height="1200" alt="S2_stress_raw_ecg" src="https://github.com/user-attachments/assets/7af9b1fd-6bf2-41f7-8525-a136fedfde95" />


This raw ECG visual inspection plot shows the unfiltered ECG signal for subject **S2 during the stress condition**. The x-axis represents time in seconds, while the y-axis represents ECG amplitude. The repeated sharp upward deflections correspond to cardiac R-peaks, which are the main reference points used later for heartbeat interval and HRV calculation.

Visual inspection is an important early quality-control step because it allows the signal to be evaluated before automated preprocessing and feature extraction. In this segment, the R-peaks are clearly visible across the recording, suggesting that the ECG contains usable cardiac information. However, the waveform also shows baseline fluctuation, amplitude variation, and smaller irregular components between the main peaks. These features may reflect movement, muscle activity, electrode contact changes, or other noise sources during the stress condition.

The plot therefore supports two important inferences. First, the signal is sufficiently structured for R-peak detection because the main heartbeat peaks are identifiable. Second, the presence of drift and irregular fluctuations indicates that filtering and artifact handling are necessary before HRV features are extracted. Without these preprocessing steps, noise could shift detected peak locations or introduce false peaks, leading to inaccurate NN intervals.

This figure helps justify the preprocessing pipeline used in the notebook. It shows why raw ECG should not be used directly for HRV analysis, even when the cardiac rhythm is visible. Instead, visual inspection confirms that the signal is usable but requires cleaning to improve the reliability of R-peak detection and downstream stress-related HRV interpretation.

**Takeaway:** The raw ECG contains clear heartbeat structure, but visible drift and noise justify filtering and artifact control before HRV analysis.


## Artifact-Marked ECG Plot

<img width="3567" height="1164" alt="S2_stress_marked_artifacts" src="https://github.com/user-attachments/assets/88acde0b-7200-454d-aa12-5964d99a638e" />


This artifact-marked ECG plot shows a filtered ECG segment from subject **S2 during the stress condition**, with suspected artifact points highlighted on top of the waveform. The blue ECG trace represents the filtered cardiac signal, while the marked points indicate samples that were identified as potential artifacts or abnormal signal regions.

The purpose of this plot is to visually confirm whether artifact detection is occurring in physiologically plausible locations. In ECG preprocessing, artifacts can arise from movement, electrode contact issues, abrupt amplitude shifts, or non-cardiac noise. If these noisy regions are not identified, they can distort R-peak detection and produce inaccurate NN intervals, which would directly affect HRV metrics.

In this segment, the artifact markers appear around high-amplitude peaks and irregular waveform sections. This suggests that the artifact-marking step is flagging parts of the signal that may influence heartbeat detection or interval estimation. Because HRV analysis depends on precise timing between consecutive normal beats, marking these segments helps protect the later feature extraction stage from contaminated data.

This figure is therefore an important quality-control step. It shows that the pipeline is not only extracting HRV features automatically, but also checking whether the ECG signal contains regions that may reduce confidence in the derived measurements. The presence of marked artifacts indicates that stress-condition ECG data may contain noise or irregular segments that should be handled carefully before interpreting HRV results.

**Takeaway:** Artifact marking helps identify potentially unreliable ECG regions, improving confidence in later R-peak detection and HRV feature extraction.

---

## Raw vs Filtered ECG Plot

<img width="3567" height="1844" alt="S2_stress_raw_vs_filtered" src="https://github.com/user-attachments/assets/c628fee8-fe5d-4502-87ee-844dfd4e8f6a" />


This figure compares the **raw ECG** and **filtered ECG** signal for subject **S2 during the stress condition**. The top panel shows the original ECG waveform before filtering, while the bottom panel shows the same segment after filtering has been applied.

In the raw ECG plot, the main cardiac peaks are visible, but the signal also contains baseline fluctuation and smaller noise components. These unwanted variations can make the ECG less stable and may interfere with accurate R-peak detection. Since HRV analysis depends on precise timing between consecutive R-peaks, even small distortions in the ECG signal can affect the reliability of the extracted NN intervals.

After filtering, the ECG waveform appears cleaner and more centered around the baseline. The sharp R-peaks remain clearly visible, while some slower drift and background noise are reduced. This suggests that the filtering step improves signal quality without removing the key cardiac information needed for heartbeat detection.

The comparison is important because it visually demonstrates that preprocessing is not simply a technical step; it directly supports the physiological inference made later in the analysis. A cleaner ECG signal allows more reliable R-peak detection, which leads to more accurate HRV features. Therefore, this plot provides evidence that the pipeline is preparing the ECG signal appropriately before extracting stress-related cardiac measures.

**Takeaway:** Filtering improves ECG signal quality by reducing noise and baseline fluctuation while preserving the R-peaks needed for HRV analysis.

---


## PSD Before and After Filtering Plot

<img width="2964" height="1464" alt="S2_stress_psd_before_after" src="https://github.com/user-attachments/assets/b76c2758-e2b0-4bf7-9b24-ba9dd07f97b1" />


This figure shows the **power spectral density (PSD) before and after filtering** for subject **S2 during the stress condition**. The x-axis represents frequency in Hz, while the y-axis shows power spectral density on a logarithmic scale. This allows the plot to show both large and very small power differences across the frequency range.

The raw ECG PSD shows that the original signal contains power across a broad range of frequencies, including higher-frequency components that may reflect noise, muscle activity, movement artifacts, or other non-cardiac signal contamination. These components can interfere with clean ECG interpretation and may reduce the reliability of later R-peak detection.

After filtering, the PSD changes substantially. The filtered ECG retains the lower-frequency signal components that are most relevant for the ECG waveform, while power is strongly reduced at higher frequencies. This indicates that the filter is suppressing unwanted frequency content rather than simply changing the signal visually in the time domain.

The drop in the filtered PSD at higher frequencies provides frequency-domain evidence that preprocessing is working as intended. In other words, the filtering step reduces noise while preserving the main cardiac rhythm information needed for heartbeat detection. This strengthens confidence that the subsequent R-peak detection and HRV feature extraction are based on a cleaner physiological signal.

The notch-like reduction around the mid-frequency range also suggests targeted attenuation of specific frequency content, likely related to line noise or filter behavior. This is important because concentrated noise at specific frequencies can distort ECG morphology and affect automated detection algorithms.

**Takeaway:** Filtering successfully reduces unwanted spectral power, especially at higher frequencies, while preserving ECG-relevant information for reliable R-peak detection and HRV analysis.





## HRV Paired Plots


The HRV paired plots compare each subject’s baseline and stress values across selected HRV metrics. In these plots, each thin line represents one participant and traces that participant’s physiological shift from the baseline condition to the stress condition.

This visualization is especially informative because it emphasizes within-subject change rather than relying only on group-level averages. Since HRV can vary substantially between individuals, paired plots help control for individual differences by showing how each participant changes relative to their own baseline.

The direction and steepness of each line provide an immediate visual inference. A downward line from baseline to stress suggests a reduction in HRV during stress, which is consistent with reduced parasympathetic regulation and increased autonomic strain. An upward or flat line may indicate individual variability, weaker stress response, noise, or possible artifact influence.

The thick black line represents the group mean trend, summarizing the overall direction of change across all subjects. If this mean line decreases from baseline to stress, it supports the project hypothesis that stress reduces HRV compared with baseline. Therefore, the paired plots do more than display values; they visually indicate whether stress produces a consistent physiological shift across participants.

---

## Mean Heart Rate Plot

<img width="1766" height="1464" alt="mean_nn_ms_baseline_vs_stress" src="https://github.com/user-attachments/assets/8b815bf0-8d60-4bab-a4cd-392a3ee18496" />

The mean heart rate paired plot shows a clear increase from baseline to stress for most subjects. Each subject’s line generally slopes upward, indicating that heart rate becomes faster during the stress condition compared with that individual’s own resting or baseline state.

This pattern supports the inference that the stress task produced a measurable cardiovascular response. Physiologically, stress activates the sympathetic branch of the autonomic nervous system, which prepares the body for increased cognitive and emotional demand. As sympathetic drive increases, cardiac activity accelerates, resulting in shorter intervals between heartbeats and a higher mean heart rate.

The consistency of the upward trend across subjects suggests that the heart rate response is not only a group-level effect but also a repeated within-subject pattern. Although some individual variation may remain, the overall direction of change indicates that stress reliably increases cardiovascular arousal.

Takeaway: Stress increases heart rate, reflecting heightened sympathetic activation during the stress condition.

---

## Mean NN Interval Plot

<img width="1766" height="1464" alt="pnn50_pct_baseline_vs_stress" src="https://github.com/user-attachments/assets/1a93996a-ae74-4bc7-8d47-5434524d8fff" />

The mean NN interval paired plot shows that most subjects have lower NN intervals during stress compared with baseline. Since each line represents the change within the same subject, the downward trend indicates that the time between consecutive normal heartbeats becomes shorter during the stress condition.

NN interval refers to the time gap between two consecutive normal R-peaks in the ECG signal. When NN intervals decrease, the heart is beating more frequently because there is less time between beats. This pattern is therefore directly consistent with the observed increase in mean heart rate.

Physiologically, this suggests a stress-related shift toward greater sympathetic activation. During stress, the body prepares for increased cognitive and emotional demand by accelerating cardiovascular activity. As a result, heartbeats occur closer together, producing shorter NN intervals and higher heart rate.

This plot is important because it confirms the heart rate finding from the opposite perspective: heart rate increases because the interval between beats decreases. The paired decrease in NN interval therefore strengthens the inference that stress produces a clear cardiac acceleration response.

Takeaway: Stress shortens the time between heartbeats, reflecting faster cardiac activity during the stress condition.

---

## SDNN Plot

<img width="1766" height="1464" alt="sdnn_ms_baseline_vs_stress" src="https://github.com/user-attachments/assets/e9f2c5a0-2268-4047-ad0c-6dfcf7e54801" />

The SDNN paired plot shows mixed subject-level changes between baseline and stress. Unlike mean heart rate and mean NN interval, the lines do not move in one consistent direction across participants. Some subjects show a decrease in SDNN during stress, while others show little change or even an increase.

SDNN represents the overall variability of normal-to-normal heartbeat intervals. A lower SDNN is often interpreted as reduced overall HRV, which may reflect decreased autonomic flexibility during stress. However, because the responses in this plot are inconsistent across subjects, the evidence for a clear stress-related SDNN reduction is weak in this notebook.

This mixed pattern suggests that SDNN may be more sensitive to individual differences, signal quality, window length, or artifact correction than the simpler heart rate and NN interval measures. It may also indicate that the stress response is not equally expressed in overall HRV variability for all subjects. Therefore, SDNN should be interpreted cautiously rather than treated as a strong standalone marker of stress in this analysis.

Takeaway: SDNN does not show a consistent stress effect in this notebook, suggesting that overall HRV variability is less clearly separated between baseline and stress than heart rate-based measures.
---

## RMSSD Plot

<img width="1766" height="1464" alt="rmssd_ms_baseline_vs_stress" src="https://github.com/user-attachments/assets/db89ed08-4b4e-483a-ae19-142940feb174" />

The RMSSD paired plot shows **variable subject-level responses** between baseline and stress. Some subjects show a reduction in RMSSD during stress, while others remain relatively unchanged or show an increase. This means that the direction of change is not uniform across participants.

RMSSD reflects short-term beat-to-beat variability and is commonly interpreted as an index of parasympathetic, or vagal, regulation. A decrease in RMSSD during stress would normally suggest reduced vagal control and lower short-term HRV. However, in this analysis, the paired lines do not show a consistent downward pattern across subjects.

This variability suggests that RMSSD may not be a strong standalone discriminator of stress in this notebook. The mixed response could reflect individual differences in stress reactivity, differences in signal quality, artifact influence, or the sensitivity of RMSSD to short analysis windows. Although the physiological expectation is that stress may reduce parasympathetic activity, the observed pattern is not consistent enough to support a reliable stress-related decrease.

Therefore, RMSSD should be interpreted cautiously in this dataset. Rather than providing strong evidence of stress-related HRV suppression, it suggests that short-term parasympathetic changes may vary across individuals or may require cleaner signals, longer windows, or additional features for clearer separation.

**Takeaway:** RMSSD does not show a statistically reliable stress-related decrease in this analysis, indicating that short-term HRV is not consistently reduced during stress across subjects.

---

## pNN50 Plot

<img width="1766" height="1464" alt="pnn50_pct_baseline_vs_stress" src="https://github.com/user-attachments/assets/4e4d399b-660e-4c2d-be44-cb2430fea0a1" />


The pNN50 paired plot shows that many subjects have **lower pNN50 values during stress** compared with baseline, although the pattern is not universal across all participants. Several paired lines move downward from baseline to stress, suggesting that stress may reduce the frequency of large beat-to-beat changes in NN intervals.

pNN50 measures the percentage of adjacent NN intervals that differ by more than 50 ms. Higher pNN50 values generally indicate greater short-term heartbeat variability, while lower values suggest more regular and less variable heartbeat timing. Therefore, a reduction in pNN50 during stress can be interpreted as a possible sign of reduced parasympathetic, or vagal, regulation.

Compared with SDNN and RMSSD, the pNN50 plot appears to show a more visible stress-related decrease for several subjects. However, the effect is not fully consistent. Some participants show little change or do not follow the expected downward trend. In addition, the Wilcoxon p-value is borderline, meaning the statistical evidence is close to significance but not strong enough to confirm a reliable group-level effect with high confidence.

This suggests that pNN50 may capture a **possible stress-related reduction in short-term HRV**, but the result should be interpreted cautiously. The finding supports the project hypothesis in a weak-to-moderate way, indicating that stress may reduce beat-to-beat variability, but the evidence is not as robust as the heart rate and mean NN interval results.

**Takeaway:** pNN50 shows a likely stress-related reduction, but the evidence is borderline rather than strongly confirmed.
---



## Binary Cross-Entropy Loss Plot


The binary cross-entropy loss plot shows how the logistic-regression model’s error changes across training epochs. The x-axis represents the training epoch, while the y-axis represents the loss value. Since binary cross-entropy measures the difference between the predicted class probabilities and the true baseline/stress labels, lower loss indicates that the model is making more accurate probability estimates over time.

The curve decreases smoothly as training progresses, suggesting that the model is successfully learning patterns from the extracted HRV features. In this context, the decreasing loss implies that the HRV feature set contains some information that helps the model distinguish between baseline and stress samples.

The smooth shape of the curve is also important from an optimization perspective. There is no clear sign of numerical instability, divergence, or strong oscillation. This indicates that the learning rate and optimization process are reasonably stable for this simple logistic-regression setup.

However, a decreasing training loss alone does not prove that the model generalizes well to unseen data. It mainly shows that the model can fit the training data in a stable way. Therefore, the loss plot should be interpreted together with validation performance, accuracy, confusion matrix, or other evaluation metrics.

**Takeaway:** The logistic-regression model is learning from the HRV features, and the training process appears numerically stable.


---

## PCA Latent-Space Plot

The PCA latent-space plot projects the HRV features into two principal components.

The x-axis represents the first principal component, and the y-axis represents the second principal component. The colors represent baseline and stress conditions.

The first principal component explains 75.7% of the variance, while the second principal component explains 15.5%. Together, these two components capture most of the variation in the HRV feature set.

The plot shows partial separation between baseline and stress, suggesting that HRV features contain condition-related structure. However, overlap remains between the two groups, meaning that baseline and stress are not perfectly separable using these features alone.

Takeaway: HRV features contain stress-related structure, but baseline and stress still overlap.

---

## t-SNE Latent-Space Plot

The t-SNE latent-space plot shows a nonlinear embedding of the HRV windows.

The axes are arbitrary t-SNE dimensions and do not have direct physiological meaning. Points that are close together have similar HRV feature profiles. Colors indicate baseline and stress conditions.

Some local regions appear enriched for one condition, suggesting that certain HRV patterns are more common during either baseline or stress.

However, t-SNE should not be overinterpreted because it can create apparent clusters depending on parameter settings and local neighborhood structure.

Takeaway: Some local clusters are condition-enriched, but the plot should be interpreted cautiously.

---

## UMAP Latent-Space Plot

The UMAP latent-space plot shows another nonlinear embedding of the HRV feature space.

The x-axis represents UMAP dimension 1, and the y-axis represents UMAP dimension 2. These axes are embedding coordinates and should not be interpreted as direct physiological measurements.

The plot shows some separation between baseline and stress points, suggesting that HRV features contain information related to stress condition. However, overlap remains between the two groups.

This overlap indicates that HRV features do not perfectly distinguish stress from baseline, likely because of individual differences and physiological variability.

Takeaway: Some condition separation exists, but HRV features do not perfectly separate stress from baseline.

---

## JAX Latent Stress Score Plot

The JAX latent stress score plot shows predicted stress probability for baseline and stress windows.

The x-axis represents condition, and the y-axis represents predicted stress probability. The boxplot summarizes the distribution of predicted probabilities, while the jittered points show individual windows.

Stress windows generally receive higher predicted stress probabilities than baseline windows. This suggests that the model learned a stress-related signal from the HRV features.

However, there is overlap between the baseline and stress distributions. Some baseline windows receive relatively high stress probabilities, and some stress windows receive lower probabilities.

Takeaway: Stress windows generally receive higher stress probabilities, but the classifier is not perfect.

---

## Overall Summary

The plots provide visual support for the stress hypothesis. The strongest and most consistent effects are observed in mean heart rate and mean NN interval. Heart rate increases during stress, while NN interval decreases, showing faster cardiac activity under stress.

Some HRV variability measures, such as pNN50, suggest a possible reduction during stress, but the evidence is weaker and less consistent. SDNN and RMSSD show mixed subject-level responses and do not demonstrate a clear stress effect.

The PCA, t-SNE, UMAP, and JAX stress score plots show that HRV features contain condition-related information. However, overlap between baseline and stress remains, indicating that HRV-based stress detection is informative but not perfectly separable.


---

## References 
1. Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection*. Proceedings of the 20th ACM International Conference on Multimodal Interaction, 400–408.
2. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). *Heart rate variability: Standards of measurement, physiological interpretation and clinical use*. Circulation, 93(5), 1043–1065.




