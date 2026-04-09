---
output:
  word_document: default
  html_document: default
editor_options: 
  markdown: 
    wrap: 72
---

# HeartCycle: A Comprehensive Dataset of Synchronized Impedance Cardiography and Echocardiography for Accurate Hemodynamic Predictions

## Authors

Eduardo Illueca Fernandez, Ricardo Couceiro, Farhad Abtahi, Jorge
Henriques, Rui Pedro Paiva, Lino Goncalves, José Millet Roig, Fernando
Seoane, Jens Muehlsteff, Paulo Carvalho

**Version:** 1.0.0

## Citation

When using this resource, please cite:

> Illueca Fernandez, E., Couceiro, R., Abtahi, F., Henriques, J., Paiva,
> R. P., Goncalves, L., Millet, J., Seoane, F., Muehlsteff, J., &
> Carvalho, P. (2025). HeartCycle: A comprehensive dataset of
> synchronized impedance cardiography and echocardiography for accurate
> hemodynamic predictions (version 1.0.0). PhysioNet. RRID:SCR_007345.

Please also include the standard citation for PhysioNet:

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C.,
> Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and
> PhysioNet: Components of a new research resource for complex
> physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
> RRID:SCR_007345.

## Abstract

The "HeartCycle" dataset offers a comprehensive collection of
synchronized impedance cardiography (ICG) and echocardiography (ECHO)
signals, supplemented with finger photoplethysmography (PPG), heart
sounds, and electrocardiography (ECG) data from 17 healthy volunteers.
Collected during the HeartCycle project (FP7–216695), this dataset aims
to address biases in the ICG waveform, particularly the ABEXYOZ complex,
where the B and X points do not precisely align with the aortic valve
opening and closing notches. The biases in B and X point detection are
critical for hemodynamic prediction because these characteristic points
are used to calculate essential diagnostic parameters including systolic
time intervals (PEP and LVET), contractility, stroke volume, and cardiac
output. By providing synchronized ICG and ECHO signals, researchers can
better understand these biases and develop more accurate models for
hemodynamic parameter computation. The dataset is stored in HDF5 format,
facilitating the storage of complex data structures and easy access to
various physiological parameters. It is ideal for developing machine
learning models to enhance the detection of characteristic points in ICG
signals. For instance, machine learning models can be used to detect
characteristic points for improved heart left ventricle ejection time
(LVET) estimation or mapping the ICG signal with the different
mechanical events in the cardiac cycle using the ECHO as a reference.
Detailed metadata and usage notes are included to support data
utilization across different software environments. Ethical approval was
obtained from the University of Coimbra Hospital's ethics committee, and
informed consent was provided by all participants.

## Background

Impedance cardiography (ICG) is one of the reference methods for
portable devices in assessing several key hemodynamic descriptors, such
as the systolic time intervals (STI) and the cardiac output (CO) [1].
The ICG principle is based on the measurement of the thorax impedance
variations (dZ/dt) that are influenced by airflow through the lungs,
blood flow from the left ventricle to the aorta and lung perfusion [2].
The assessment of the systolic time intervals requires the determination
of the ICG’s characteristic points, which are assumed to be correlated
to the opening and closing of the aortic valve [3]. The waveform
obtained from the dZ/dt signal presents the ABEXYOZ complex, where B
correspond to the aortic valve opening notch and X to the aortic valve
closing notch [4]. However, previous studies conclude there is a bias in
the ICG waveform, and B and X points do not exactly fit with the notches
[5]. While previous datasets - as the  ReBeatICG database  [6] - have
typically provided ICG measurements synchronized with ECG, the
simultaneous acquisition of multiple modalities remains unexplored in
open access resources. For this reason, this dataset provides
researchers ICG signals synchronized with echocardiography record (ECHO)
to understand the bias present in the ICG waveforms, and it proposes new
models and methods to correct this bias. To the best of our knowledge,
this is the first publicly available dataset offering simultaneous ICG,
ECHO, ECG, and PPG recordings, enabling comprehensive multi-modal
analysis of cardiac hemodynamics and validation of ICG-derived
parameters against the gold-standard ECHO measurements.

## Methods

The data were extracted from physiological studies conducted during the
HeartCycle project over healthy subjects. This dataset stores data from
17 volunteers. 

The HDF5 data files record the synchronized signals for impedance
cardiography (ICG), finger photoplethysmography (PPG), heart sounds and
echocardiography (ECHO). For each one of these modalities, the
synchronized signal for electrocardiography (ECG) was also provided. In
addition, data files containing the hemodynamic and physiological
parameters computed for each record were included. The MATLAB Software
[7] was used to process signals and to generate the synchronized HDF5
files. A detailed description of the content of the HDF5 files is
provided in the FileMetadata.csv files.

### Equipment Used

The ICG and ECG signals were recorded using Niccomo
® (TotalMedicalSolutions, Netherlands). Data were exported in .txt
format.  Vivid Ultrasound from General Electric was used to record ECHO
data, and the data were processed using DICOM software, which created
images in M-mode and Doppler mode. The ECHO output is stored as image in
the \_091 group as a three dimensions array where the first dimension
correspond to the channel, the second to the time and the third
represents depth or distance from the transducer in M-Mode and velocity
in Doppler Mode.  For PPG, the sensors from Philips ® V26 Patient
Monitor were used to collect the signals. Last, a Meditron stethoscope
was used to annotate heart sounds.

Sampling rates depend on the device and the synchronization procedure.
For Niccomo, sampling rate is equal to 200 for ECG and ICG signals; for
Vivid Ultrasound the sampling rate is 136 for ECHO signals and the
synchronized ECG; for the V26 Patient Monitor the sampling rate is 500
for the ECG signal and PPG signal; and a sampling rate equal to 44100
was used for phnocardiography and the synchronized ECG. All sampling
rates are documented in the *Rate* group inside each one of the groups
in the HDF5 files.

### Synchronization Protocol

The synchronization protocol for handling and processing data in the
HeartCycle dataset includes acquisition, organization, and annotation of
physiological signals (ECG, ICG, HS, PPG and ECHO). The acquisition
process involves recording data using each hardware specific software,
generating files that are then copied to a designated directory
structure based on acquisition date and volunteer ID. These raw
acquisition files are processed to produce multiple CSV files, which are
later imported into MATLAB for further processing. Each acquisition is
assumed to generate different signal segments for each one of the
modalities, corresponding to a different record as outlined in the
acquisition protocol.

Once imported, these signals are organized into MATLAB files named after
each volunteer. These files contain three primary structures: aq_info
(acquisition details like date and location), vol_info (volunteer
demographics and health status), and measure (a matrix organizing ECG,
ICG, HS, PPG and ECHO with different hemodynamic parameters collected).
Each cell in the measure matrix includes time vectors, signal data,
labels, sampling rates, units, run identifiers, and descriptions of the
volunteer’s activity during that run. Manual annotation of PPG signals
was required, based on visual inspection and protocol-defined intervals,
to ensure accurate interpretation and segmentation of physiological
responses during each activity.

## Data Description

The dataset comprises 2.3 GB detailed recordings from healthy subjects.
The files are systematically named to reflect the subject ID, the date
(randomized) and the record id. For instance, the file
`CH07_59146237_s0000029.h5` correspond to the record s0000029 from the
subject CH07 and performed on the day 59146237.

The H5 format allows to store complex data structures as the one
presented in this dataset. The structure of this file is summarized in
the table below. Each column represents each one of the medical devices
used, and in each cell a vector or matrix with the corresponding data is
stored.

### Data Structure Tables

In concrete, there are a total of 208 records stored in HDF5 files and
distributed in three experiments. There are 32 records in the experiment
folder 59146237, 84 records for the experiment folder 59146238 and 92
records for the experiment folder 59146239. The subject distribution is
presented in *Table 1*.

#### Table 1: Subject Distribution and Demographics

| Subject ID | Age | Height (cm) | Weight (kg) | Gender | BMI | Experiment Folder | Number of Records |
|---------|---------|---------|---------|---------|---------|---------|---------|
| CHC01 | 20 | 181 | 68 | M | 20.76 | 59146238 | 27 |
| CHC02 | 19 | 155 | 52 | F | 21.63 | 59146238 | 15 |
| CHC03 | 24 | 175 | 76 | M | 24.82 | 59146238 | 10 |
| CHC04 | 20 | 170 | 60 | F | 20.76 | 59146238 | 11 |
| CHC05 | 19 | 154 | 47 | F | 19.81 | 59146238 | 10 |
| CHC06 | 19 | 171 | 62 | M | 21.20 | 59146238 | 10 |
| CHC07 | 40 | 179 | 76 | M | 23.72 | 59146238 | 9 |
| CHC08 | 19 | 170 | 63 | M | 21.80 | 59146238 | 8 |
| CHC09 | 29 | 170 | 92 | M | 31.83 | 59146238 | 12 |
| CHC10 | 24 | 167 | 61 | M | 21.87 | 59146238 | 8 |
| CHC11 | 28 | 182 | 77 | M | 23.25 | 59146239 | 6 |
| CHC12 | 20 | 181 | 74 | M | 22.59 | 59146239 | 7 |
| CHC13 | 19 | 179 | 78 | M | 24.34 | 59146239 | 30 |
| CHC14 | 21 | 170 | 85 | M | 29.41 | 59146239 | 14 |
| CHC15 | 21 | 172 | 72 | M | 24.34 | 59146239 | 17 |
| CHC16 | 20 | 178 | 77 | M | 24.30 | 59146239 | 11 |
| CHC17 | 21 | 174 | 70 | M | 23.12 | 59146239 | 14 |

The HDF5 format allows to store complex data structures as the one
presented in this dataset. The structure of this file is summarized in
Table 3. Each column represents each one of the medical devices used,
and in each cell a vector or matrix with the corresponding data is
stored. For instance, ICG data can be accessed at C[4,2] in the HDF5
array - index can vary in function of the programming language. Please
note PPG data is only present in the experiment 59146237, so this signal
is not recorded in all HDF5 files.

#### Table 2: Sampling Rates by Device and Modality

| **Niccomo** | **Stethoscope** | **Echocardiogram** | **PPG** |
|-----------------|------------------|--------------------|-----------------|
| Electrocardiogram<br>Sampling Rate = 200 | Electrocardiogram<br>Sampling Rate = 44100 | Electrocardiogram<br>Sampling Rate = 136 | Electrocardiogram<br>Sampling Rate = 500 |
| Impedance<br>Sampling Rate = 200 | Phonocardiography<br>Sampling Rate = 44100 | Echocardiography<br>Sampling Rate = 136 | Plethysmography<br>Sampling Rate = 125 |
| Time of the R peaks of the ECG | Time of the R peaks of the ECG | Time of the R peaks of the ECG | Time of the R peaks of the ECG |
| Time of aortic valve opening | Time of aortic valve opening | Time of aortic valve opening | Time of aortic valve opening |
| Pre-ejection period | Pre-ejection period | Pre-ejection period | Pre-ejection period |
| Time of aortic valve closure | Time of aortic valve closure | Time of aortic valve closure | Time of aortic valve closure |
| Left ventricle ejection time | Left ventricle ejection time | Left ventricle ejection time | Left ventricle ejection time |

*Table 3* provide a more detailed mapping between .h5 group IDs and the
physiological signals/devices, which is also documented in the README.md
and GroupMapping.csv. Most of the signals are stored as a 2 dimensional
array of shape (1, time), while AVO and AVC are a 1-dimensional array
with the time coordinates of the event, and PEP and LVET includes the
time interval in milliseconds. However, the echo-related group \_091 has
the shape (3, time, distance/velocity), which differs from the other
signals. This three-dimensional array represent three echocardiography
signals in three device channels. The last dimension depend on the
echocardiography mode, as some files include M-Mode and other files
Doppler Mode. For a clearer interpretation, we recomend to split the
array in three matrix and compute the transposed matrix to have the time
in X-axis.

#### Table 3: HDF5 File Structure - Data Fields

| ID | Signal | Units | Dim | ID | Signal | Units | Dim | ID | Signal | Units | Dim | ID | Signal | Units | Dim |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| \_030 | ECG | mV | 2 | \_060 | ECG | mV | 2 | \_090 | ECG | mV | 2 | \_120 | ECG | \- | 2 |
| \_031 | IMP | Ohm | 2 | \_061 | PCG | s | 2 | \_091 | ECHO | \- | 3 | \_121 | PPG | s | 2 |
| \_032 | RPEAKS | s | 2 | \_062 | RPEAKS | s | 2 | \_092 | RPEAKS | s | 2 | \_122 | RPEAKS | s | 2 |
| \_033 | AVO | \- | 2 | \_063 | AVO | \- | 1 | \_093 | AVO | \- | 1 | \_123 | AVO | \- | 1 |
| \_034 | PPEjec | ms | 2 | \_064 | PEP | ms | 1 | \_094 | PEP | ms | 1 | \_124 | PEP | ms | 1 |
| \_035 | AVC | \- | 2 | \_065 | AVC | \- | 1 | \_095 | AVC | \- | 1 | \_125 | AVC | \- | 1 |
| \_036 | LVET | ms | 2 | \_066 | LVET | ms | 1 | \_096 | LVET | ms | 1 | \_126 | LVET | ms | 1 |

Last, some physiological parameters are also recorded from Niccomo, as
specified in *Table 4*.

#### Table 5: HDF5 File Structure - Hemodynamic Parameters from Niccomo Device

| ID    | Signal        | Units               | Dim |
|-------|---------------|---------------------|-----|
| \_000 | Event         | \-                  | 2   |
| \_001 | SPO2          | \-                  | 2   |
| \_002 | O/C           | \%                  | 2   |
| \_003 | Load          | W                   | 2   |
| \_004 | HPD           | ms                  | 2   |
| \_005 | DC            | 1/min               | 2   |
| \_006 | TFC           | 1/kOhm              | 2   |
| \_007 | FC            | 1/min               | 2   |
| \_008 | Heather       | Ohm/s²              | 2   |
| \_009 | Z<sm>0</sm>   | Ohm                 | 2   |
| \_010 | QI-ICG        | \%                  | 2   |
| \_011 | AV Interval   | ms                  | 2   |
| \_012 | DBP           | mmHg                | 2   |
| \_013 | PAM           | mmHg                | 2   |
| \_014 | SBP           | mmHg                | 2   |
| \_015 | PAWP          | mmHg                | 2   |
| \_016 | CVP           | mmHg                | 2   |
| \_017 | ETR           | \%                  | 2   |
| \_018 | STR           | \-                  | 2   |
| \_019 | SVR           | dyn·s·cm<SM>-5</SM> | 2   |
| \_020 | SpO<sm>2</sm> | \%                  | 2   |
| \_021 | LCW           | kg\*m               | 2   |
| \_022 | VE            | ml                  | 2   |
| \_023 | SVRI          | dyn·s·cm<SM>-5</SM> | 2   |
| \_024 | IC            | m²                  | 2   |
| \_025 | ACI           | l/min/m²            | 2   |
| \_026 | DO<sm>2</sm>I | 1/100/s²            | 2   |
| \_027 | IEjecI        | ml/min/m²           | 2   |
| \_028 | IV            | ml/m²               | 2   |
| \_029 | LCWI          | 1/1000/s            | 2   |

**Legend:** - SPO2: Oxygen Saturation - O/C: Opening/Closing Ratio -
HPD: Hemodynamic Parameter Duration - DC: Duty Cycle - TFC: Thoracic
Fluid Content - FC: Frequency/Cardiac - QI-ICG: Quality Index ICG - DBP:
Diastolic Blood Pressure - PAM: Pulmonary Artery Mean Pressure - SBP:
Systolic Blood Pressure - PAWP: Pulmonary Artery Wedge Pressure - CVP:
Central Venous Pressure - ETR: Ejection Time Ratio - STR: Systolic Time
Ratio - SVR: Systemic Vascular Resistance - SVRI: Systemic Vascular
Resistance Index - LCW: Left Cardiac Work - LCWI: Left Cardiac Work
Index - VE: Ventricular Ejection - IC: Index Cardiac - ACI: Acceleration
Index - IEjecI: Index Ejection Index - ECG: Electrocardiogram - IMP:
Impedance - PCG: Phonocardiography - ECHO: Echocardiography - PPG:
Photoplethysmography - RPEAKS: R peaks of the ECG - AVO: Aortic Valve
Opening - PEP: Pre-Ejection Period - AVC: Aortic Valve Closure - LVET:
Left Ventricular Ejection Time - Dim: Dimensionality of the data array

### Dataset Structure

The dataset is composed of three experiments: - `59146237` -
`59146238` - `59146239`

Each experiment is stored in directories with the same name. In each
directory, there is a subdirectory called `measure` which contains the
H5 files with the data. Two additional files are in each experiment
directory: - `FileMetadata.csv` - `SubjectMetadata.csv` -
`SubjectMetadata.md`

## Usage Notes

This dataset provides ICG recordings with echocardiography as reference,
as well as other techniques, suitable for developing machine learning
models to detect the real notches and improve the accuracy of
hemodynamic parameter computation from ICG. To utilize the data,
researchers can use different data science environments for reading HDF5
data, as Jupyter, R Studio or MATLAB - among others. In consequence,
this dataset is not software dependent.

The traceability between subjects, files and experiments is specified in
the SubjectMetadata.csv file, where the demographic data of each subject
is also summarized. In addition, data quality was also included for each
one of the record files and spe. It was measured as the synchronization
percentage between two physiological signals or datasets (e.g., ICG and
ECHO), and it is defined as the proportion of temporally aligned data
points or valid overlapping segments relative to the total expected
duration of synchronization, expressed as a , where represents the
fraction of temporally misaligned or invalid data segments relative to
the total recording duration.

While this dataset offers valuable multi-modal synchronized recordings,
researchers should note certain limitations. The relatively small sample
size may limit generalization across diverse populations, and the
controlled laboratory acquisition conditions may not fully represent
real-world clinical or ambulatory settings. For this reason, we
encourage researchers to use this dataset from a data science
perspective for training new AI models, but we recommend avoiding the
extraction of physiological conclusions that cannot be extrapolated to
other populations 

Further details about how to use and how to get started with the dataset
can be found in the README.md file. Furthermore, the script
tutorial.py includes some examples on how loading HDF5 data.

```{python}
import h5py 
import numpy as np 
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')

print(f['measure']['value'].keys())

ecg = f['measure']['value']['_030']['value']['data']['value'][0,:] 
time = f['measure']['value']['_030']['value']['time']['value'][0,:]

plt.figure(figsize=(12, 5)) 
plt.plot(time,ecg) 
plt.title("ECG signal") 
plt.xlabel('Time (ms)') 
plt.ylabel('ECG (mV)') 
plt.show()
```

It is important to note tha Niccomo impedance signal stored in the HDF5
file is the raw signal. For most applications, the derivative dZ/dt is
required. An easy way to compute this derivative in Python is as
follows, where icg_time is the array with the timestamp and icg_record
is the array with the raw ICG signal.

```{python}
import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')


print(f['measure']['value'].keys())

icg = f['measure']['value']['_031']['value']['data']['value'][0,:]
time = f['measure']['value']['_031']['value']['time']['value'][0,:]

dt = np.mean(np.diff(time))
dz = np.gradient(icg, dt)
```

For echocardiography, a special preprocessing is required to load and
visualize the image matrix.

```{python}
import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('./59146237/measure/CH07_59146237_s0000029.h5', 'r')


echo = f['measure']['value']['_091']['value']['data']['value'][0,:,:].transpose()

plt.figure(figsize=(12, 5))
plt.imshow(echo, cmap='viridis', aspect='auto')
plt.title("Echocardiography Image")
plt.xlabel('Time (ms)')
plt.show()
```

## Ethics

The study was approved by the University of Coimbra Hospital's ethics committee under the
reference CES-238 and fully complies with the Declaration of Helsinki.

## Conflicts of Interest

The authors declare no conflict of interest.

## References

1.  Kubicek, W. G., Patterson, R. P., & Witsoe, D. A. (1970). Impedance
    cardiography as a noninvasive method of monitoring cardiac function
    and other parameters of the cardiovascular system. Annals of the New
    York Academy of Sciences, 170(2), 724-732.

2.  Visser, K. R., Mook, G. A., Van der Wall, E., & Zijlstra, W. G.
    (1993). Theory of the determination of systolic time intervals by
    impedance cardiography. Biological psychology, 36(1-2), 43-50.

3.  Chan, G. S., Middleton, P. M., Celler, B. G., Wang, L., &
    Lovell, N. H. (2007). Automatic detection of left ventricular
    ejection time from a finger photoplethysmographic pulse oximetry
    waveform: comparison with Doppler aortic measurement. Physiological
    measurement, 28(4), 439.

4.  Benouar, S., Hafid, A., Attari, M., Kedir-Talha, M., & Seoane, F.
    (2018). Systematic variability in ICG recordings results in ICG
    complex subtypes–steps towards the enhancement of ICG
    characterization. Journal of electrical bioimpedance, 9(1), 72.

5.  Carvalho, P., Paiva, R. P., Henriques, J., Antunes, M., Quintal, I.,
    & Muehlsteff, J. (2011, January). Robust characteristic points for
    ICG-definition and comparative analysis. In International Conference
    on Bio-inspired Systems and Signal Processing (Vol. 2, pp. 161-168).
    SCITEPRESS
    
6.  Pale U, Meier D, Muller N, Arza A, Atienza D. ReBeatICG database. Zenodo; 2021. https://doi.org/10.5281/zenodo.4725433

7.  The MathWorks Inc. (2022). MATLAB version: 9.13.0 (R2022b), Natick,
    Massachusetts: The MathWorks Inc. <https://www.mathworks.com>

------------------------------------------------------------------------

### Access Policy

Anyone can access the files, as long as they conform to the terms of the
specified license.

### License

Open Data Commons Attribution License v1.0

### Topics

impedance cardiography, echocardiography, cardiovascular physiology,
machine learning, electrophysiological study

### Corresponding Author

Eduardo Illueca Fernandez\
Department of Clinical Science, Intervention and Technology\
Karolinska Institutet\
Stockholm 17177, Sweden\
Email: [eduardo.illueca\@ki.se](mailto:eduardo.illueca@ki.se){.email}
