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

This workflow supports the inference that stress-related changes in autonomic regulation can be observed  in raw ECG morphology and also in the variability patterns derived from consecutive R-peaks. By comparing HRV windows across experimental conditions, the pipeline provides a structured way to examine whether baseline and stress states produce separable physiological signatures.

The NeuroPype implementation was designed to mirror the same logic in a more modular signal-processing environment. An important implementation insight concerns frequency-domain visualization. SpectrumPlot does not operate directly on raw ECG or HRV time-series data; it requires frequency-axis spectral input. Therefore, it must be preceded by a spectral estimation node such as Spectrogram or WelchSpectrum. Similarly, HeartRate and HeartRateVariability nodes must receive the output stream from RDetection, specifically the stream ending in peaks, because HR and HRV calculations depend on detected R-peak timing rather than the raw ECG waveform itself.

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


This raw ECG visual inspection plot shows the unfiltered ECG signal for subject **S2 during the stress condition**. The x-axis represents time in seconds, while the y-axis represents ECG amplitude. The repeated sharp upward deflections correspond to cardiac R-peaks, which are the main reference points used for heartbeat interval and HRV calculation.

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

 Therefore, this plot provides evidence that the pipeline is preparing the ECG signal appropriately before extracting stress-related cardiac measures.

**Takeaway:** Filtering improves ECG signal quality by reducing noise and baseline fluctuation while preserving the R-peaks needed for HRV analysis.

---


## R-Peak Detection Check

<img width="3567" height="1164" alt="S2_stress_rpeaks" src="https://github.com/user-attachments/assets/a0a04f25-b7c0-4e66-b6fb-94dfdf1db34e" />


This R-peak detection check shows the cleaned ECG signal with the detected R-peaks marked as circular points. The continuous waveform represents the cleaned ECG, while each marker identifies a heartbeat location selected by the detection algorithm.

This plot is important because R-peak detection is the foundation of the entire HRV workflow. HRV features are not calculated directly from the ECG waveform itself; they are derived from the timing differences between consecutive detected R-peaks. Therefore, the quality of R-peak detection directly determines the reliability of the NN intervals and all later HRV metrics.

In this segment, the detected markers align closely with the sharp upward deflections of the ECG waveform. This suggests that the algorithm is identifying the main cardiac peaks rather than random noise or smaller waveform fluctuations. The spacing between detected peaks also appears physiologically plausible, which supports confidence that the resulting NN interval sequence reflects real heartbeat timing.

This figure provides a visual quality-control checkpoint before HRV extraction. If peaks were missed, falsely added, or placed at incorrect locations, the HRV results could be distorted. Here, the alignment between markers and ECG peaks indicates that the cleaned signal is suitable for interval calculation and downstream stress analysis.

**Takeaway:** The R-peak detection appears reliable in this ECG segment, supporting accurate NN interval calculation and trustworthy HRV feature extraction.



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



## Logistic Regression Training Loss Plot

<img width="2367" height="1164" alt="jax_training_loss" src="https://github.com/user-attachments/assets/5d5b085a-b6a3-4949-a7c6-aeedd02101d4" />



This plot shows the binary cross-entropy loss of the logistic-regression model across training epochs. The x-axis represents the number of training epochs, while the y-axis represents the model’s prediction error. Because binary cross-entropy measures how far the predicted probabilities are from the true baseline and stress labels, a lower value indicates better model fit.

The loss decreases sharply during the early epochs, showing that the model quickly learns an initial separation pattern from the HRV features. After this rapid drop, the curve continues to decline more gradually and begins to flatten. This suggests that the model is still improving, but the largest learning gains occur early in training.

The smooth downward shape of the curve indicates stable optimization. There is no visible sign of divergence, sudden spikes, or strong oscillation, which means the learning process is numerically well behaved. This supports the interpretation that the selected learning rate and logistic-regression setup are appropriate for the feature set.

From an inference perspective, the decreasing loss suggests that the extracted HRV features contain information that helps distinguish baseline from stress samples. However, the training loss alone only confirms that the model is fitting the training data. It should be interpreted together with validation accuracy, test performance, or classification metrics before making a strong claim about generalization.

**Takeaway:** The logistic-regression model learns a stable separation pattern from the HRV features, with the strongest improvement occurring early in training.



---

## PCA Latent-Space Visualization of HRV Features

<img width="2067" height="1764" alt="pca_hrv_latent_space" src="https://github.com/user-attachments/assets/07f52c1d-8c42-43ac-8508-a0128885f9b3" />


This PCA latent-space plot visualizes the HRV feature windows in a reduced two-dimensional space. Each point represents one HRV feature window, with blue points corresponding to baseline samples and orange points corresponding to stress samples. The purpose of this plot is to examine whether the extracted HRV features form different patterns for baseline and stress conditions.

The plot shows a visible separation tendency between the two conditions. Baseline samples are more concentrated in the lower region of the space, especially around negative values on Dimension 2, while stress samples are more widely distributed and extend strongly into higher values on Dimension 2. This suggests that stress-related HRV windows occupy a different region of the feature space compared with baseline windows.

At the same time, the two classes are not perfectly separated. There is noticeable overlap near the center of the plot, where both baseline and stress samples appear together. This indicates that HRV features contain stress-relevant information, but the distinction is not absolute. Some stress windows resemble baseline physiology, and some baseline windows fall near stress-like regions. This overlap may reflect individual variability, transitional physiological states, artifact influence, or limitations of using only time-domain HRV features.

The upward spread of the stress class is especially informative. It suggests that stress does not simply shift all samples in one uniform direction; instead, it increases variability in the feature representation. In physiological terms, this may indicate that stress responses differ across subjects or across time windows, with some windows showing stronger autonomic changes than others.

Because PCA is a linear dimensionality-reduction method, this figure should be interpreted as an exploratory visualization rather than definitive classification evidence. However, the partial separation between baseline and stress supports the broader inference that ECG-derived HRV features contain meaningful information about stress-related autonomic changes.

**Takeaway:** PCA shows partial separation between baseline and stress HRV windows, suggesting that the extracted HRV features contain stress-relevant structure, although the overlap indicates that the classes are not perfectly separable using these features alone.


---


## t-SNE Latent-Space Visualization of HRV Features

<img width="2067" height="1764" alt="tsne_hrv_latent_space" src="https://github.com/user-attachments/assets/1d1c2b91-4895-4004-9d60-d947125ea844" />


This t-SNE latent-space plot shows the HRV feature windows projected into a two-dimensional nonlinear space. Each point represents one HRV window, with blue points indicating baseline samples and orange points indicating stress samples. Unlike PCA, which captures linear structure, t-SNE is designed to emphasize local neighborhood relationships, meaning points that appear close together have similar HRV feature patterns.

The plot shows several visible clusters and curved bands, suggesting that the HRV windows contain structured physiological patterns rather than random scatter. Some regions are dominated by baseline samples, especially in the lower and right-side areas, while several stress-dominant groups appear in the upper-left and central regions. This indicates that stress and baseline windows show partially different local feature relationships.

However, the separation is not complete. There are multiple overlapping regions where baseline and stress samples appear close together. This overlap suggests that some stress windows resemble baseline-like physiology, while some baseline windows share features with stress-like windows. Such overlap may reflect individual variability, transitional autonomic states, window-level noise, or limitations of using HRV features alone.

Compared with the PCA visualization, t-SNE reveals more local clustering and nonlinear structure. This suggests that the baseline-versus-stress distinction may not be fully captured by a simple linear projection, but may exist more clearly in local or nonlinear patterns within the HRV feature space.

From an inference perspective, this plot supports the idea that ECG-derived HRV features carry stress-relevant information, but also shows that stress classification is not perfectly separable. Therefore, the model may learn useful patterns, but performance will likely depend on feature quality, artifact control, subject variability, and the chosen classifier.

**Takeaway:** t-SNE shows meaningful local clustering of HRV windows, with partial baseline–stress separation, but the overlap indicates that stress detection from HRV is informative rather than perfectly distinct.


---

## Latent-Space Visualization of HRV Features

<img width="2067" height="1764" alt="umap_hrv_latent_space" src="https://github.com/user-attachments/assets/4b785d49-c446-40a1-80fa-e3adc57a4d3f" />


This UMAP latent-space plot shows the HRV feature windows projected into a two-dimensional nonlinear space. Each point represents one HRV window, with blue points indicating baseline samples and orange points indicating stress samples. UMAP is useful here because it preserves both local neighborhood structure and broader manifold relationships, helping reveal whether baseline and stress windows occupy different regions of the HRV feature space.

The plot shows a clearer spatial organization than a simple scatter of random points. Baseline samples are concentrated mainly on the left and lower-central regions, while stress samples extend more strongly toward the right side of the plot. The orange stress-dominant clusters on the right suggest that some stress windows have HRV patterns that are distinct from most baseline windows.

At the same time, the separation is not complete. Several baseline and stress points overlap near the center-left region, indicating that some windows share similar HRV characteristics across conditions. This overlap suggests that stress-related physiology is not uniform across all subjects or time windows. Some stress windows may resemble baseline states, while some baseline windows may contain higher arousal or noisy segments.

Compared with PCA, the UMAP visualization reveals more nonlinear structure in the data. The curved clusters and separated branches suggest that stress-related differences may not lie along a single linear axis. Instead, the HRV features appear to form a more complex manifold where condition-related patterns emerge in specific regions of the space.

From an inferential perspective, this plot supports the idea that ECG-derived HRV features contain stress-relevant information. The presence of stress-dominant regions suggests that the model has meaningful structure to learn from. However, because the classes still overlap, HRV alone may provide partial rather than perfect separation between baseline and stress.

**Takeaway:** UMAP shows meaningful nonlinear structure in the HRV feature space, with stress samples forming distinct regions but still overlapping with baseline in some areas.


---

##  Latent Stress Score

<img width="1767" height="1464" alt="jax_latent_stress_score" src="https://github.com/user-attachments/assets/0df24366-777d-42e2-8a68-ea6117e380b4" />

This plot shows the predicted stress probability produced by the logistic-regression model for baseline and stress samples. The y-axis represents the model’s learned stress score, where values closer to 0 indicate a stronger baseline-like prediction and values closer to 1 indicate a stronger stress-like prediction.

The baseline samples are mostly concentrated at low predicted stress probabilities, with the median near the lower part of the scale. This suggests that the model generally assigns baseline windows a low likelihood of being stress. In contrast, the stress samples are shifted upward, with many points and the boxplot median located at much higher predicted probabilities. This indicates that the model has learned a meaningful separation between the two conditions.

The spread of the points is also important. Baseline contains some higher predicted probabilities, and stress contains some lower predicted probabilities, showing that the separation is not perfect. These overlapping predictions may reflect subject-level variability, noisy HRV windows, artifact effects, or physiological periods where stress and baseline features are similar. However, the overall upward shift in the stress distribution suggests that the HRV features contain stress-relevant information.

From an inference perspective, this figure is stronger than the training loss plot because it shows how the learned model scores each condition after training. The clear difference between the baseline and stress distributions indicates that the classifier is not merely optimizing numerically; it is learning a condition-related pattern in the HRV feature space.

**Takeaway:** The JAX logistic-regression model assigns higher stress probabilities to stress samples than to baseline samples, suggesting that the extracted HRV features contain meaningful stress-related information.


---

## Overall Summary

The plots provide visual support for the stress hypothesis. The strongest and most consistent effects are observed in mean heart rate and mean NN interval. Heart rate increases during stress, while NN interval decreases, showing faster cardiac activity under stress.

Some HRV variability measures, such as pNN50, suggest a possible reduction during stress, but the evidence is weaker and less consistent. SDNN and RMSSD show mixed subject-level responses and do not demonstrate a clear stress effect.

The PCA, t-SNE, UMAP, and JAX stress score plots show that HRV features contain condition-related information. However, overlap between baseline and stress remains, indicating that HRV-based stress detection is informative but not perfectly separable.


---

## References 
1. Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection*. Proceedings of the 20th ACM International Conference on Multimodal Interaction, 400–408.
2. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). *Heart rate variability: Standards of measurement, physiological interpretation and clinical use*. Circulation, 93(5), 1043–1065.




