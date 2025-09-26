
# NFS-Fi: A Wi-Fi Near-Field Sensing Dataset for Multi-Person Activity Recognition

Welcome to NFS-Fi dataset!
You can download the NFS-Fi dataset from the following link: www.zenodo.com.

We warmly welcome researchers to utilize our dataset and greatly value your feedback, which will help us further improve and enrich this resource.
To the best of our knowledge, this is the first dataset that enables Wi-Fi-based multi-person activity recognition utilizing near-field sensing.
We kindly request that you cite our paper "[**Paper**](xxx)" if you use this dataset.

## Contents
[1. Introduction](#1-introduction) 

[2. Dataset Description](#2-dataset-description)

[3. Usage Instructions](#3-usage-instructions)


## 1. Introduction

Our NFS-Fi dataset is a practical Wi-Fi multi-person sensing dataset, offering three key advantages over existing datasets.

- First, leveraging diverse physical information, the multi-link near-field sensing strategy enables practical multi-person sensing beyond simple scenarios or mere reliance on neural network fitting.

- Second, the dataset contains organic traffic from the normal operation of smart devices, without injecting evenly spaced sensing packets, thus avoiding interference with default communication and reflecting realistic conditions.

- Third, the dataset is built using up-to-date NICs compliant with IEEE 802.11ac/ax standards, keeping Wi-Fi sensing aligned with the latest technological developments.


## 2. Dataset Description

### 2.1. Data Collection
Please refer to **Section IV-A** of our [paper](xxx) for a detailed description of the experiment setup.
In this section, we provide comprehensive instructions that enable users to collect their own data.


### 2.2. Dataset Overview
Our NFS-Fi dataset contains 64,823 samples, collected from 56 subjects performing 10 activities across 6 distinct environments.
For detailed statistics, please see Table 1.

<br/>

**<center>Table 1: NFS-Fi Dataset Overview</center>**

![Table1](\Images\Table.png)

<br/>

Each data collection involves 2–4 subjects performing the activities simultaneously, and each subject combination is included in two different environments. The specific subject assignments are detailed as follows:

- Environments 0 and 1
  + Case A: Subjects 0 and 1.
  + Case B: Subjects 2, 3, and 4.
  + Case C: Subjects 5, 6, and 7.
  + Case D: Subjects 8, 9, 10, and 11.
  + Case E: Subjects 12, 13, 14, and 15.

- Environments 2 and 3
  + Case A: Subjects 16 and 17.
  + Case B: Subjects 18 and 19.
  + Case C: Subjects 20 and 21.
  + Case D: Subjects 22, 23, and 24.
  + Case E: Subjects 25, 26, and 27.
  + Case F: Subjects 28, 29, 30, and 31.
  + Case G: Subjects 32, 33, 34, and 35.

- Environments 4 and 5
  + Case A: Subjects 36 and 37.
  + Case B: Subjects 38 and 39.
  + Case C: Subjects 40 and 41.
  + Case D: Subjects 42, 43, and 44.
  + Case E: Subjects 45, 46, and 47.
  + Case F: Subjects 48, 49, 50, and 51.
  + Case G: Subjects 52, 53, 54, and 55.



### 2.3. Dataset Statistics
In our NFS-Fi dataset, Subject 1 contributes the most valid activities (2,563) and Subject 49 the fewest (624).
Among all activities, PP has the most samples (6,780), while TP has the fewest (6,075).
Across the six environments, LR contains the most samples (12,121) and CR the fewest (9,735).
The detailed distribution is shown in Figure. 1.


![Fig_sample](\Images\stat_sample.png)

**<center>Figure 1: Statistics of samples across subjects, activities, and environments.</center>**

<br/>

Furthermore, on average, each sample contains 77 CSI entries, with a maximum time interval (Max. TI) of approximately 0.25 s and an average data collection duration of 1.86 s.
The CSI entry statistics for each sample are presented in Figure. 2.

<div style="display: flex; justify-content: space-between; align-items: center; gap: 15px;">
    <img src="/Images/stat_number.png" alt="Fig_sample" style="width: 30%;">
    <img src="/Images/stat_maxinter.png" alt="Fig_result" style="width: 30%;">
    <img src="/Images/stat_duration.png" alt="Fig_chart" style="width: 30%;">
</div>

**<center>Figure 2: Statistics of CSI entries across samples.</center>**

<br/>

### 2.4. Dataset Structure

#### 2.4.1. Our NFS-Fi dataset is organized into two primary directories:
- ```Raw_Data```: Contains the raw, unprocessed data.
  + Data: Raw CSI data.
    * num_point: Number of CSI entries.
    * max_interval: Maximum time interval between CSI entries (in seconds).
    * mean_interval: Average time interval between CSI entries (in seconds).
    * spanning: Total duration of the data collection (in seconds).
    * csi: CSI entries.
    * rssi: RSSI entries.
    * time: Timestamp.
    * flag_cond_satisfied: A flag indicating whether the data passed predefined quality checks.
  + Domain: Domain index, which corresponds to the subject index.
  + Label: Activity index
  + Scena: Environment index.
- ```Proc_Data```: Contains the pre-processed data, which is ready to be used as input for neural networks. The specific processing pipeline is described in the following [section](#33-workflow). This serves as an example; users can also develop their own processing algorithms.
  + data: Processed CSI sequences, formatted for direct input into neural network models for activity recognition.
  + label: Activity index
  + domain: Domain (subject) index
  + scen: Environment index.
  + max_seq_length: The length of the processed sequences.

#### 2.4.2. Index Description
- Label
  + 0: Push\&Pull (PP).
  + 1: Sweeping (SW).
  + 2: Drawing circle (DC).
  + 3: zig\&zag (ZZ).
  + 4: Phone typing (TP).
  + 5: Handshaking (HS).
  + 6: Bending (BD).
  + 7: Jumping (JP).
  + 8: Rotating (RT).
  + 9: Walking (WK).
- Scena
  + 0: Meeting room (MR).
  + 1: Lecture room (LR).
  + 2: Discussion room (DR).
  + 3: Classroom (CR).
  + 4: Office room (OR).
  + 5: Self-study room (SR).
    


## 3. Usage Instructions

To facilitate easy access to and usage of our NFS-Fi dataset, we provide a demonstration script.

### 3.1. Environment and Hardware (optional)

- Ubuntu 22.04.2 LTS
- 5.15.0-94-generic kernel
- Python 3.8.16
- CUDA Version: 12.2.
- NVIDIA RTX A5000
- MATLAB R2023b

### 3.2. Install

- Install Python: Please ensure Python 3.8.16 is installed on your computer. You can also download the Python source from the [official website](https://www.python.org/).

- Set Up Virtual Environment: It is recommended to set up a virtual environment to ensure a clean and isolated environment for the implementation. Tools like ```conda``` can be used for this purpose. Make sure to activate your virtual environment before proceeding.

- Install the necessary packages: We provide the requirements.txt in source code. You can install them by ```pip install -r requirements.txt```.

- MATLAB: The MATLAB code is implemented in R2023b version without relying on any specialized toolboxes, ensuring compatibility with earlier versions.

### 3.3. Workflow

- Data Processing:
  + We have provided pre-processed data, which is saved in the ```Proc_data``` directory. For details, please refer to ```main_Process.m``` in the folder '''Process'''. Users can adapt this code by modifying lines **32** and **39** to implement their own methods.

- Running the Neural Network Model:
  + Open ```main_demo.py``` and change the ```FILE_PATH``` on line **34** to your own ```Proc_data``` directory path. Then, execute the script. We have included detailed comments in the code to facilitate user customization and debugging.
