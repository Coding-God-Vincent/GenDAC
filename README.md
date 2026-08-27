# Feasibility-First Generative Diffusion-Based Deep Reinforcement Learning for Inter-Slice Radio Resource Management in RAN Slicing

## 1. Environment
* **OS**: Ubuntu  24.04.2 LTS
* **Python**: 3.12.3
* **PyTorch**: 2.9.1+cu128

## 2. Installation
### 2.1 Clone this project
```bash
cd ~
git clone https://github.com/Coding-God-Vincent/GenDAC.git
cd GenDAC
```

### 2.2 Create new conda environment
```bash
# Create environment with Python 3.12.3
conda create -n GenDAC_venv python=3.12.3 -y
```

### 2.3 Use existing conda environment
```bash
# Activate environment
conda activate GenDAC_venv

# Set Matplotlib backend to Agg 
conda env config vars set MPLBACKEND=Agg 

# Reactivate the environment to apply the environment variable 
conda deactivate 
conda activate GenDAC_venv

# Install Pytorch 2.9.1 with CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```

## 3. Repository structure
```bash
├── CSV_generator_github/             # Scripts for exporting TensorBoard event data to CSV files
│   └── csv_generator1.py             # Convert selected TensorBoard scalar metrics into CSV files
│
├── Env/                              # Downlink RAN slicing simulation environments
│   ├── __init__.py                   # Environment package initialization
│   ├── env_fixedUE.py                # Single-BS RAN slicing environment with fixed UEs
│   └── env_movingUE.py               # Single-BS RAN slicing environment with UE mobility
│
├── Graph_generator_github/           # Scripts for generating evaluation figures from CSV results
│   ├── 6_algos.py                    # Compare GenDAC with GAN-DDQN, LSTM-A2C, Hard Slicing, PPO, and SAC
│   ├── GenDAC_MlpAC.py               # Compare diffusion actor with MLP actor
│   ├── bits_packets.py               # Compare packet-bit and packet-count state representations
│   ├── bound.py                      # Sensitivity analysis of action-logit clipping bounds
│   ├── denoise_step.py               # Sensitivity analysis of diffusion denoising steps
│   ├── lambda.py                     # Sensitivity analysis of reconstruction-loss weight
│   ├── more_state.py                 # Compare original and extended state representations
│   ├── nr_oriented.py                # Compare six algorithms under the NR-oriented scenario
│   ├── proportion.py                 # Plot inter-slice resource-allocation proportions
│   ├── reward_function.py            # Compare different reward-function designs
│   ├── throughput.py                 # Compare system throughput and spectral efficiency
│   └── w_wo_rec.py                   # Compare GenDAC with and without reconstruction loss
│
├── Logs_github/                      # TensorBoard logs generated during experiments
│   └── <algorithm>/                  # Algorithm name, e.g., GenDAC, PPO, SAC, etc.
│       └── exp*/                     # Individual experiment
│           └── tensorboard/          # TensorBoard event files
│
├── Outcome_github/                   # Experimental data and final thesis figures
│   ├── CSVs/                         # Experimental CSV data used for the thesis figures
│   │   └── seed_{124,125,126,127,128}/ # Results corresponding to different random seeds
│   │       └── <algorithm>_csv/      # Results of an algorithm or experimental setting
│   │           ├── utility.csv       # System utility over decision windows
│   │           ├── se.csv            # Spectral efficiency over decision windows
│   │           ├── qoe_*.csv         # Slice Satisfaction Rate (SSR) results
│   │           ├── action_*.csv      # Inter-slice resource-allocation actions
│   │           ├── throughput.csv    # System throughput over decision windows
│   │           └── *.csv             # Other recorded experiment metrics
│   │
│   ├── CSVs_new/                     # CSV results exported from newly generated experiments
│   │   └── seed_<seed>/              # Automatically created directory for each random seed
│   │       └── <algorithm>_csv/      # Automatically created directory for each experiment
│   │           └── *.csv             # Exported TensorBoard scalar metrics
│   │
│   └── Figures/                      # Final evaluation figures used in the thesis
│       ├── 6_algos/                  # Main comparison among six algorithms
│       ├── GenDAC_MlpAC/             # Diffusion-actor vs. MLP-actor comparison
│       ├── bound/                    # Action-logit clipping-bound sensitivity results
│       ├── denoise_step/             # Diffusion denoising-step sensitivity results
│       ├── lambda/                   # Reconstruction-loss weight sensitivity results
│       ├── more_state/               # Extended-state ablation results
│       ├── nr_oriented/              # NR-oriented scenario comparison results
│       ├── proportion/               # Inter-slice resource-allocation proportion figures
│       ├── reward_function/          # Reward-function comparison results
│       ├── state/                    # State-representation ablation results
│       ├── throughput/               # Throughput comparison results
│       └── w_wo_rec/                 # Reconstruction-loss ablation results
│
├── Temp_Figures/                     # Figures generated during individual training experiments
│   └── <algorithm>/                  # Algorithm name
│       └── exp*/                     # Individual experiment
│           ├── QoE.png               # Slice Satisfaction Rate learning curves
│           ├── SE.png                # Spectral-efficiency learning curve
│           └── Utility.png           # System-utility learning curve
│
├── Test_Figures/                     # Figures newly generated by scripts in Graph_generator_github
│   └── <experiment>/                 # Figure-generation experiment, e.g., 6_algos
│       └── *.{pdf,svg,png}           # Generated evaluation figures
│
├── Utils/                            # Algorithm-specific models, optimizers, buffers, and helper functions
│   ├── Diffusion_utils/              # Diffusion actor and GenDAC optimization utilities
│   │   ├── __init__.py               # Diffusion utility package initialization
│   │   ├── D2AC_model.py             # Generative diffusion model and double-critic networks
│   │   ├── D2AC_opt.py               # GenDAC actor-critic optimization and parameter updates
│   │   ├── diffusion.py              # Diffusion policy and reverse denoising process
│   │   └── helpers.py                # Diffusion-related helper and noise utilities
│   │
│   ├── GAN_utils/                    # Supporting utilities for the GAN-DDQN baseline
│   │   ├── ReplayMemory.py           # Experience replay memory
│   │   ├── RolloutStorage.py         # Rollout data storage
│   │   ├── data_structures.py        # Supporting data structures
│   │   ├── hyperparameters.py        # GAN-DDQN hyperparameter definitions
│   │   ├── plot.py                   # Plotting utilities
│   │   ├── utils.py                  # General GAN-DDQN helper functions
│   │   └── wrappers.py               # Environment and data wrapper utilities
│   │
│   ├── LSTM_A2C_utils/               # Supporting utilities for the LSTM-A2C baseline
│   │   └── utils.py                  # Action-space, state-processing, and reward utilities
│   │
│   ├── MlpAC_utils/                  # Utilities for the MLP actor-critic ablation
│   │   ├── Model.py                  # MLP actor and double-critic networks
│   │   └── MlpAC_opt.py              # MLP actor-critic optimization and parameter updates
│   │
│   ├── MlpAC_clamp_rec_utils/        # Alternative MLP utilities for clamp/reconstruction experiments
│   │   ├── ActionTransform.py        # Action-transformation utilities
│   │   ├── Model.py                  # Actor and critic model definitions
│   │   └── MlpAC_opt.py              # Optimization and parameter-update logic
│   │
│   ├── PPO_utils/                    # Supporting utilities for the PPO baseline
│   │   ├── Model.py                  # PPO actor-critic model
│   │   ├── PPOopt.py                 # PPO optimization and policy-update logic
│   │   └── RolloutBuffer.py          # On-policy rollout buffer
│   │
│   ├── SAC_utils/                    # Supporting utilities for the SAC baseline
│   │   ├── Model.py                  # SAC actor and critic models
│   │   ├── ReplayBuffer.py           # Off-policy experience replay buffer
│   │   └── SACopt.py                 # SAC optimization and parameter-update logic
│   │
│   └── seed.py                       # Set random seeds for reproducible experiments
│
├── GenDAC.py                         # Main training script for the proposed GenDAC method
├── GANDDQN.py                        # GAN-DDQN baseline implementation
├── Hard_Slicing.py                   # Hard-slicing baseline with fixed inter-slice resource allocation
├── LSTM_A2C.py                       # PyTorch implementation of the LSTM-A2C baseline
├── MlpAC.py                          # MLP actor-critic ablation replacing the diffusion actor
├── PPO.py                            # PPO baseline implementation
├── SAC.py                            # SAC baseline implementation
├── __init__.py                       # Project package initialization
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

## 4. Usage Examples

Since the configuration procedure is similar for all algorithms, `GenDAC.py` is used as an example when explaining the training-script configuration.

To demonstrate the complete experimental workflow consistently, Sections 4–7 use the main six-algorithm comparison as a running example. The six algorithms are GenDAC, GAN-DDQN, LSTM-A2C, Hard Slicing, PPO, and SAC.

For each algorithm, five independent experiments are performed using the following experiment names and random seeds:

| Experiment Name | Random Seed |
|---|---:|
| `exp1` | 124 |
| `exp2` | 125 |
| `exp3` | 126 |
| `exp4` | 127 |
| `exp5` | 128 |

Therefore, each algorithm is evaluated using five independent runs, resulting in 30 runs in total for the six-algorithm comparison.

The same experiment–seed mapping is maintained throughout the training, TensorBoard logging, CSV export, and figure-generation procedures described in the following sections.

### 4.1 Configure Experiment Names and Random Seeds

In `GenDAC.py`, replace **Lines 285–286** with the desired experiment names and random seeds.

For the six-algorithm comparison used as the running example in this README, five independent experiments are performed for each algorithm:

```python
exps = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
seeds = [124, 125, 126, 127, 128]
```

Each experiment name in `exps` corresponds to the random seed at the same position in `seeds`:

```text
exp1 → seed 124
exp2 → seed 125
exp3 → seed 126
exp4 → seed 127
exp5 → seed 128
```

The number and order of entries in `exps` and `seeds` must be the same.

The same experiment–seed mapping should be configured for all six algorithms so that their results can later be compared using the same set of random seeds.

For the baseline algorithms, modify the corresponding lines shown below:

| Algorithm | Experiment Name | Random Seed |
|---|---:|---:|
| `GenDAC.py` | Line 285 | Line 286 |
| `GANDDQN.py` | Line 55 | Line 51 |
| `LSTM_A2C.py` | Line 28 | Line 24 |
| `PPO.py` | Line 23 | Line 24 |
| `SAC.py` | Line 29 | Line 31 |
| `Hard_Slicing.py` | Line 14 | Line 15 |

#### Record the Experiment Information

Before running the experiments, the algorithm name, experiment name, and random seed used for each run should be recorded.

These three pieces of information together identify an experimental run:

```text
Algorithm + Experiment Name + Random Seed
```

For the running example used in this README, the experiment records are:

| Algorithm | Experiment Names | Random Seeds |
|---|---|---|
| GenDAC | `exp1`–`exp5` | 124–128 |
| GAN-DDQN | `exp1`–`exp5` | 124–128 |
| LSTM-A2C | `exp1`–`exp5` | 124–128 |
| Hard Slicing | `exp1`–`exp5` | 124–128 |
| PPO | `exp1`–`exp5` | 124–128 |
| SAC | `exp1`–`exp5` | 124–128 |

For every algorithm, the experiment–seed mapping is:

```text
exp1 → seed 124
exp2 → seed 125
exp3 → seed 126
exp4 → seed 127
exp5 → seed 128
```

**This mapping should be retained throughout the complete experimental workflow.**


### 4.2 Configure the Network Scenario

The environment file used in this study is `Env/env_movingUE.py`, which provides two network scenarios: the **4G LTE scenario** and the **5G NR scenario**. For the main differences between the two scenarios, please refer to [Scenario Comparison](URL).

For `GenDAC.py`, modify **Line 302**:

```python
nr_oriented_scenario = False
```

uses the **4G LTE scenario**, while:

```python
nr_oriented_scenario = True
```

uses the **5G NR scenario**.

For the baseline algorithms, modify the corresponding line shown below:

| Algorithm | `nr_oriented_scenario` |
|---|---:|
| `GenDAC.py` | Line 302 |
| `GANDDQN.py` | Line 59 |
| `LSTM_A2C.py` | Line 32 |
| `PPO.py` | Line 29 |
| `SAC.py` | Line 36 |
| `Hard_Slicing.py` | Line 20 |



After configuring `exps`, `seeds`, and `nr_oriented_scenario` settings, run the desired algorithm from the project root directory:

```bash
# Our Method
python GenDAC.py

# Baseline algorithms
python GANDDQN.py
python LSTM_A2C.py
python Hard_Slicing.py
python PPO.py
python SAC.py
```

Each seed is trained independently using its corresponding experiment name.

During training, the TensorBoard event files for each experiment are stored under:

```text
Logs_github/<algorithm>/<experiment>/tensorboard/
```

For example, after running the five GenDAC experiments in the running example, the TensorBoard logs are stored under:

```text
Logs_github/GenDAC/exp1/tensorboard/
Logs_github/GenDAC/exp2/tensorboard/
Logs_github/GenDAC/exp3/tensorboard/
Logs_github/GenDAC/exp4/tensorboard/
Logs_github/GenDAC/exp5/tensorboard/
```

The same directory structure is used for the other algorithms. For example, the PPO experiments are stored under:

```text
Logs_github/PPO/exp1/tensorboard/
Logs_github/PPO/exp2/tensorboard/
Logs_github/PPO/exp3/tensorboard/
Logs_github/PPO/exp4/tensorboard/
Logs_github/PPO/exp5/tensorboard/
```

After the training of each seed is completed, several core training metrics are automatically plotted and saved as PNG files under:

```text
Temp_Figures/<algorithm>/<experiment>/
```

These figures provide a quick visualization of the training process for important performance metrics, such as:

- Slice Satisfaction Rate (SSR)
- Spectral Efficiency (SE)
- System utility

For example:

```text
Temp_Figures/
└── GenDAC/
    ├── exp1/
    │   ├── QoE.png
    │   ├── SE.png
    │   └── Utility.png
    │
    ├── exp2/
    │   ├── QoE.png
    │   ├── SE.png
    │   └── Utility.png
    │
    ├── exp3/
    │   ├── QoE.png
    │   ├── SE.png
    │   └── Utility.png
    │
    │
    ├── exp4/
    │   ├── QoE.png
    │   ├── SE.png
    │   └── Utility.png
    │
    │
    └── exp5/
        ├── QoE.png
        ├── SE.png
        └── Utility.png
```

The required directories under `Temp_Figures/` are created automatically during execution if they do not already exist. Therefore, there is no need to manually create the output directories before running the training scripts.

These PNG files are intended for quickly inspecting the training behavior of each individual run. For more detailed analysis and figure generation across multiple seeds, the TensorBoard logs can be exported to CSV files as described in Section 6 and processed using the scripts in `Graph_generator_github/` as described in Section 7.

## 5. Real-Time Training Monitoring with TensorBoard

TensorBoard can be used to monitor the training progress in real time. The required TensorBoard package is already included in `requirements.txt`.

### 5.1 Start Training

Using GenDAC as an example, start the training script in one terminal:

```bash
python GenDAC.py
```

During training, TensorBoard event files are automatically generated under:

`Logs_github/<algorithm>/<experiment>/tensorboard`

For example, the TensorBoard logs of GenDAC follow the structure:

`Logs_github/GenDAC/<experiment>/tensorboard`

### 5.2 Monitor Training Progress

While the training script is running, open another terminal in the project root directory and run TensorBoard for the corresponding experiment:

```bash
tensorboard --logdir Logs_github/GenDAC/<experiment>/tensorboard
```

Replace `<experiment>` with the experiment name defined in the training script. For example:

```bash
tensorboard --logdir Logs_github/GenDAC/exp1/tensorboard
```
In the running example, `exp1` corresponds to seed `124`. To monitor another GenDAC run, replace `exp1` with the corresponding experiment name (`exp2`–`exp5`).

The same procedure can be used for the other algorithms by replacing `GenDAC` with the corresponding algorithm directory name.

TensorBoard will provide a local URL, typically: `http://localhost:6006/`

Open the URL in a web browser to monitor the training metrics while the experiment is running. The displayed curves can be refreshed during training to inspect the latest progress.


### 5.3 Training Metrics

The **Scalars** tab in TensorBoard provides various training and system metrics, including:

* Actor, policy, reconstruction, and critic losses
* Slice Satisfaction Rate (SSR) for VoLTE, eMBB, and URLLC
* Spectral Efficiency (SE)
* System utility and reward
* Inter-slice resource allocation actions
* Action logits
* Average queue length of each slice
* System throughput

These metrics can be used to observe the learning behavior and performance of the algorithm throughout the training process.

## 6. Export TensorBoard Logs to CSV

The TensorBoard event files generated during training can be converted into CSV files using:

`CSV_generator_github/csv_generator1.py`

The script extracts selected scalar metrics from a TensorBoard event file and stores each metric as an individual CSV file.

Newly exported CSV files are stored under:

```text
Outcome_github/CSVs_new/
```

The directories under `CSVs_new/` are automatically created by `csv_generator1.py` if they do not already exist.

Note that:

- `Outcome_github/CSVs/` contains the experimental CSV data used to generate the figures reported in the thesis.
- `Outcome_github/Figures/` contains the final figures used in the thesis.
- `Outcome_github/CSVs_new/` is used to store CSV files exported from newly generated experimental results.

### 6.1 Configure the CSV Export

Before running `CSV_generator_github/csv_generator1.py`, specify the TensorBoard metrics to be exported.

In `CSV_generator_github/csv_generator1.py`, replace **Line 77** with the desired TensorBoard scalar tags.

For example:

```python
target_tags = ['qoe/volte', 'qoe/embb_general', 'qoe/urllc', 'se', 'utility', 'action/volte', 'action/embb_general', 'action/urllc', 'throughput']
```

Each tag corresponds to a scalar recorded in the TensorBoard event file. Only the metrics included in `target_tags` will be exported to CSV files.

Next, configure the random seeds and the algorithm/result name by replacing **Lines 90–91**.

For the five GenDAC experiments used in the running example:

```python
seeds = [124, 125, 126, 127, 128]
algo_name = "GenDAC"
```

The value of `algo_name` is used as the name of the output CSV directory and should correspond to the algorithm or experimental setting represented by the TensorBoard event files.

The generated CSV files follow the directory structure:

```text
Outcome_github/CSVs_new/seed_<seed>/<algo_name>_csv/
```

Therefore, the five GenDAC experiments in the running example are exported to:

```text
Outcome_github/CSVs_new/
├── seed_124/
│   └── GenDAC_csv/
├── seed_125/
│   └── GenDAC_csv/
├── seed_126/
│   └── GenDAC_csv/
├── seed_127/
│   └── GenDAC_csv/
└── seed_128/
    └── GenDAC_csv/
```

The output directories are created automatically if they do not already exist.

The number and order of entries in `seeds` must be consistent with the TensorBoard event file paths specified in `event_path_moving`, as described in Section 6.2.

### 6.2 Specify the TensorBoard Event Files

The TensorBoard event files to be exported are specified using `event_path_moving`.

TensorBoard event files generated during training can be found under:

```text
Logs_github/<algorithm>/<experiment>/tensorboard/
```

Inside each corresponding `tensorboard/` directory, locate the TensorBoard event file whose name starts with:

```text
events.out.tfevents
```

For the five GenDAC experiments in the running example, configure:

```python
seeds = [124, 125, 126, 127, 128]
algo_name = "GenDAC"
```

and specify the corresponding TensorBoard event files in the same order:

```python
event_path_moving = [
    'Logs_github/GenDAC/exp1/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp2/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp3/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp4/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp5/tensorboard/events.out.tfevents...'
]
```

Both relative paths and absolute paths can be used. When using relative paths, the paths should be relative to the directory from which the CSV generator is executed. In the examples in this README, `csv_generator1.py` is executed from the project root directory.

The order of `seeds` and `event_path_moving` must correspond to the recorded experiment–seed mapping:

```text
seed 124 → GenDAC exp1 TensorBoard event file
seed 125 → GenDAC exp2 TensorBoard event file
seed 126 → GenDAC exp3 TensorBoard event file
seed 127 → GenDAC exp4 TensorBoard event file
seed 128 → GenDAC exp5 TensorBoard event file
```

Therefore, the number of entries in `seeds` and `event_path_moving` must be the same, and their order must correspond to each other.

The same procedure is repeated for the other five algorithms by changing `algo_name` and replacing the TensorBoard event paths with the corresponding algorithm directories.

### 6.3 Generate the CSV Files

After configuring the target metrics, random seed, algorithm name, and TensorBoard event path, run the CSV generator from the project root directory:

```bash
python CSV_generator_github/csv_generator1.py
```

Each TensorBoard scalar specified in `target_tags` is exported as an individual CSV file.

The `/` character in a TensorBoard tag is replaced with `_` in the corresponding CSV filename.

For example:

```text
qoe/volte           → qoe_volte.csv
qoe/embb_general    → qoe_embb_general.csv
qoe/urllc           → qoe_urllc.csv
action/volte        → action_volte.csv
action/embb_general → action_embb_general.csv
action/urllc        → action_urllc.csv
se                  → se.csv
utility             → utility.csv
throughput          → throughput.csv
```

Each generated CSV file contains two columns:

```text
Step,Value
```

where `Step` represents the training decision window and `Value` represents the corresponding TensorBoard scalar value.

---
For the complete six-algorithm comparison used in the running example, repeat the CSV-export procedure for GenDAC, GAN-DDQN, LSTM-A2C, Hard Slicing, PPO, and SAC.

Each algorithm uses the same five random seeds and experiment–seed mapping:

```text
exp1 → seed 124
exp2 → seed 125
exp3 → seed 126
exp4 → seed 127
exp5 → seed 128
```

After the TensorBoard logs of all six algorithms have been exported, the CSV directory structure is:

```text
Outcome_github/CSVs_new/
├── seed_124/
│   ├── GenDAC_csv/
│   ├── GANDDQN_csv/
│   ├── LSTM_A2C_csv/
│   ├── Hard_Slicing_csv/
│   ├── PPO_csv/
│   └── SAC_csv/
│
├── seed_125/
│   ├── GenDAC_csv/
│   ├── GANDDQN_csv/
│   ├── LSTM_A2C_csv/
│   ├── Hard_Slicing_csv/
│   ├── PPO_csv/
│   └── SAC_csv/
│
├── seed_126/
│   ├── GenDAC_csv/
│   ├── GANDDQN_csv/
│   ├── LSTM_A2C_csv/
│   ├── Hard_Slicing_csv/
│   ├── PPO_csv/
│   └── SAC_csv/
│
├── seed_127/
│   ├── GenDAC_csv/
│   ├── GANDDQN_csv/
│   ├── LSTM_A2C_csv/
│   ├── Hard_Slicing_csv/
│   ├── PPO_csv/
│   └── SAC_csv/
│
└── seed_128/
    ├── GenDAC_csv/
    ├── GANDDQN_csv/
    ├── LSTM_A2C_csv/
    ├── Hard_Slicing_csv/
    ├── PPO_csv/
    └── SAC_csv/
```

These CSV files can then be used directly as the input of `Graph_generator_github/6_algos.py`, as described in Section 7.

---

## 7. Generate Figures from CSV Results

The scripts in `Graph_generator_github/` are used to read experimental CSV files and generate evaluation figures.

Each script corresponds to a specific experiment, comparison, or ablation study. The CSV data required to reproduce the figures reported in the thesis are already provided under:

```text
Outcome_github/CSVs/
```

All figure-generation scripts are also preconfigured to use the corresponding experimental data included in this repository. Therefore, no additional configuration is required to reproduce the provided results. Each script can be executed directly, and the generated figures will be automatically saved to its predefined directory under:

```text
Test_Figures/
```

The configuration instructions in Section 7.1 are provided for users who want to modify the existing figure-generation settings in the future, such as changing the algorithms or experimental settings to be compared, changing the number of random seeds, using newly generated CSV results, or modifying the figure output directory.

Although the number of compared algorithms or experimental settings and the plotted metrics may differ between scripts, all scripts in `Graph_generator_github/` follow the same general configuration procedure. Therefore, `Graph_generator_github/6_algos.py` is used as the representative example in the following section.

The example continues the six-algorithm and five-seed comparison introduced in Sections 4–6:

```text
Graph_generator_github/6_algos.py
```

This script generates the main comparison figures for GenDAC, GAN-DDQN, LSTM-A2C, Hard Slicing, PPO, and SAC using random seeds `124`, `125`, `126`, `127`, and `128`.

### 7.1 Modify the Figure-Generation Settings



#### 7.1.1 Configure the Algorithm Names

In `Graph_generator_github/6_algos.py`, **Lines 41–46** specify the names of the six algorithms whose CSV directories will be loaded:

```python
algo_name1 = "GenDAC"
algo_name2 = "GANDDQN"
algo_name3 = "LSTM_A2C"
algo_name4 = "Hard_Slicing"
algo_name5 = "PPO"
algo_name6 = "SAC"
```

These names must match the corresponding `<algo_name>_csv` directory names generated in Section 6:

```text
GenDAC_csv/
GANDDQN_csv/
LSTM_A2C_csv/
Hard_Slicing_csv/
PPO_csv/
SAC_csv/
```

The legend labels displayed in the generated figures are defined in **Line 49**:

```python
labels = ["GenDAC", "GAN-DDQN", "LSTM-A2C", "Hard Slicing", "PPO", "SAC"]
```

Modify the legend labels only if different names should be displayed in the generated figures.

#### 7.1.2 Configure the Random Seeds

The random seeds used to load the experimental results are specified in **Line 51**:

```python
seeds = [124, 125, 126, 127, 128]
```

These are the same five random seeds used throughout the running example in Sections 4–6.

According to the recorded experiment–seed mapping:

```text
seed 124 → exp1
seed 125 → exp2
seed 126 → exp3
seed 127 → exp4
seed 128 → exp5
```

For each random seed, `6_algos.py` loads the corresponding CSV results of all six algorithms.

#### 7.1.3 Configure the CSV Input Path

The current `6_algos.py` is configured to read the experimental CSV files used to generate the figures reported in the thesis from:

```text
Outcome_github/CSVs/
```

The CSV input path is specified separately for System Utility, Spectral Efficiency (SE), and SLA Satisfaction Rate (SSR):

```text
Line 74  → System Utility
Line 125 → Spectral Efficiency (SE)
Line 186 → SLA Satisfaction Rate (SSR)
```

The current setting at each of these lines is:

```python
csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
```

To generate figures using the newly generated experimental results from Sections 4–6, replace **Lines 74, 125, and 186** with:

```python
csv_path = Path("Outcome_github/CSVs_new") / f"seed_{seeds[j]}"
```

All three lines must be modified so that Utility, SE, and SSR are loaded from the same CSV directory.

For the running example, when `seeds[j] = 124`, the script reads the six algorithm directories under:

```text
Outcome_github/CSVs_new/seed_124/
├── GenDAC_csv/
├── GANDDQN_csv/
├── LSTM_A2C_csv/
├── Hard_Slicing_csv/
├── PPO_csv/
└── SAC_csv/
```

The same procedure is repeated automatically for seeds `125`, `126`, `127`, and `128`.

Therefore, the algorithm names, random seeds, and CSV input paths configured in `6_algos.py` must be consistent with the CSV files generated in Section 6.

> **Note:** The algorithm names, random seeds, CSV paths, and other experiment-specific settings currently defined in `Graph_generator_github/` are configured for the experimental results used in the thesis. When generating figures from newly obtained experimental results, modify these settings according to the corresponding experiment.

#### 7.1.4 Configure the Figure Output

The generated figures are stored separately from the final thesis figures.

In `Graph_generator_github/6_algos.py`, the root directory for newly generated figures is specified in **Line 39**:

```python
Figure = "Test_Figures"
```

The subdirectory for the six-algorithm comparison is specified in **Line 57**:

```python
image_path = Path(f"{Figure}/6_algos")
```

Therefore, with the default settings, the generated figures are stored under:

```text
Test_Figures/6_algos/
```

The output directory is automatically created by the script if it does not already exist. Therefore, `Test_Figures/` and `Test_Figures/6_algos/` do not need to be created manually.

To change the root output directory, replace **Line 39**.

For example:

```python
Figure = "My_Figures"
```

To change the experiment-specific subdirectory, replace **Line 57**.

For example:

```python
image_path = Path(f"{Figure}/My_Experiment")
```

Other scripts in `Graph_generator_github/` follow the same general workflow and generate figures under their corresponding experiment directories.

For example:

```text
Test_Figures/
├── 6_algos/
├── nr_oriented/
├── throughput/
└── ...
```

`Test_Figures/` is intended for newly generated figures.

In contrast:

```text
Outcome_github/Figures/
```

contains the final figures used in the thesis.

---
### 7.2 Generate the Figures

With the default settings provided in this repository, no additional configuration is required to reproduce the provided figures.

For example, to generate the main six-algorithm comparison figures, run the following command from the project root directory:

```bash
python Graph_generator_github/6_algos.py
```

With the default configuration, `6_algos.py` reads the experimental CSV data from:

```text
Outcome_github/CSVs/
```

and processes the results of the six algorithms using the five random seeds:

```text
6 algorithms × 5 random seeds = 30 experimental runs
```

The generated figures are automatically stored under:

```text
Test_Figures/6_algos/
```

The other scripts in `Graph_generator_github/` are also preconfigured with their corresponding experimental data and output directories. Therefore, they can be executed directly in the same way.

For example:

```bash
python Graph_generator_github/nr_oriented.py
python Graph_generator_github/throughput.py
python Graph_generator_github/reward_function.py
```

The generated figures will be automatically saved to the corresponding subdirectories under `Test_Figures/`.


---

## 8. Configure Ablation Studies

This section describes how to modify `GenDAC.py` to reproduce or extend two ablation studies: the effect of the reconstruction loss and the effect of different reward-function designs.

The experimental results used for these ablation studies in the thesis are already included under `Outcome_github/CSVs/`, and the corresponding figure-generation scripts are:

```text
Graph_generator_github/w_wo_rec.py
Graph_generator_github/reward_function.py
```

Therefore, these figure-generation scripts can be executed directly using the provided experimental results. The following instructions describe the modifications required in `GenDAC.py` when rerunning the ablation experiments.

### 8.1 Ablation Study of Reconstruction Loss

This ablation study evaluates the effect of the reconstruction loss by comparing GenDAC with and without the reconstruction-loss term during actor optimization.

In `GenDAC.py`, whether the reconstruction loss is included in the actor loss is controlled by **Line 451**:

```python
with_rec_loss = True
```

#### GenDAC with Reconstruction Loss

The default GenDAC configuration includes the reconstruction loss:

```python
with_rec_loss = True
```

With this setting, the actor loss includes both the policy loss and the weighted reconstruction loss.

#### GenDAC without Reconstruction Loss

To remove the reconstruction loss from actor optimization, replace **Line 451** with:

```python
with_rec_loss = False
```

With this setting, the actor is optimized without the reconstruction-loss term while the remaining GenDAC configuration is unchanged.

The provided experimental results use the following result names:

```text
GenDAC           → with reconstruction loss
GenDAC_lam_0     → without reconstruction loss
```

These names correspond to the settings used in:

```text
Graph_generator_github/w_wo_rec.py
```

After obtaining and exporting the corresponding experimental results as described in Section 6, the ablation figures can be generated using:

```bash
python Graph_generator_github/w_wo_rec.py
```

The generated figures are automatically stored under:

```text
Test_Figures/w_wo_rec/
```

### 8.2 Ablation Study of Reward Functions

This ablation study evaluates the effect of the reward-function design while keeping the GenDAC learning algorithm unchanged.

The provided experiment compares the following four reward functions:

```text
1. Feasibility-first reward
2. LSTM-A2C reward
3. GAN-DDQN reward
4. Weighted sum of SE and SSR
```

Only one `cal_reward()` function should be enabled in `GenDAC.py` for each experiment.

#### 8.2.1 Feasibility-First Reward

The default GenDAC reward function is defined in **Lines 222–237**:

```python
def cal_reward(qoe, se, qoe_weights, se_weight, SLA_threshold= 0.95, reward_clipping= False):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]

    qoe_score = np.matmul(qoe_weights, qoe.reshape((3, 1)))[0] / 10.0
    qoe_slack = (
        max(0, SLA_threshold - qoe[0])
        + max(0, SLA_threshold - qoe[1])
        + max(0, SLA_threshold - qoe[2])
    )

    qoe_penalty = 0.0

    se_base_score = (se_weight * se[0]) / 10.0
    decay = 10
    se_discount = math.exp(-decay * qoe_slack)

    reward = qoe_score - qoe_penalty + (se_base_score * se_discount)
    reward = np.array([reward])

    return utility, reward, qoe_slack, (se_base_score * se_discount)
```

This is the default reward function used by GenDAC. Therefore, no modification is required when evaluating the feasibility-first reward.

When using this reward function, keep the alternative reward functions in **Lines 247–273** commented out.

#### 8.2.2 LSTM-A2C Reward

To evaluate GenDAC using the LSTM-A2C reward function, first comment out the default reward function in **Lines 222–237**.

Then uncomment the LSTM-A2C reward function in **Lines 260–273**:

```python
def cal_reward(
    qoe,
    se,
    qoe_weights= [1, 1, 1],
    se_weight= 0.01,
    SLA_threshold= 0.95,
    reward_clipping= False
):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se[0]

    if qoe[1] >= 0.98 and qoe[0] >= 0.98:
        if qoe[2] >= 0.95:
            if se[0] < 280:
                reward = 4
            else:
                reward = 4 + (se[0] - 280) * 0.1
        else:
            reward = (qoe[2] - 0.7) * 10
    else:
        reward = -5

    reward = np.array([reward])

    return utility, reward, 0, 0
```

The remaining GenDAC settings are kept unchanged.

#### 8.2.3 GAN-DDQN Reward

To evaluate GenDAC using the GAN-DDQN reward function, first comment out the default reward function in **Lines 222–237**.

Then uncomment the reward function in **Lines 247–257**:

```python
def cal_reward(qoe, se, qoe_weights, se_weight, SLA_threshold= 0.95, reward_clipping= False):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se

    if reward_clipping:
        threshold1 = 6.5
        threshold2 = 4.5

        if utility >= threshold1:
            reward = 1
        elif utility < threshold1 and utility > threshold2:
            reward = 0
        else:
            reward = -1

        reward = np.array([reward])
    else:
        reward = utility

    return utility, reward, _, 0
```

For the GAN-DDQN reward, keep **Line 748** as:

```python
utility, reward, qoe_slack, se_part = cal_reward(
    qoe= qoe,
    se= se,
    qoe_weights= qoe_weights,
    se_weight= se_weight,
    SLA_threshold= SLA_threshold,
    reward_clipping= True
)
```

Setting `reward_clipping = True` enables the clipped reward used for this comparison.

#### 8.2.4 Weighted Sum of SE and SSR

The weighted-sum experiment uses the same `cal_reward()` function in **Lines 247–257** as the GAN-DDQN reward experiment.

However, replace **Line 748** so that:

```python
utility, reward, qoe_slack, se_part = cal_reward(
    qoe= qoe,
    se= se,
    qoe_weights= qoe_weights,
    se_weight= se_weight,
    SLA_threshold= SLA_threshold,
    reward_clipping= False
)
```

With `reward_clipping = False`, the reward is directly calculated using the weighted system utility:

```text
Reward = weighted sum of SE and SSR
```

Therefore, the difference between the GAN-DDQN reward and the weighted-sum reward in this ablation study is:

| Reward Function | `reward_clipping` |
|---|---|
| GAN-DDQN reward | `True` |
| Weighted sum of SE and SSR | `False` |

The four result names used by the provided figure-generation script are:

| Reward Function | Result Name |
|---|---|
| Feasibility-first reward | `GenDAC` |
| LSTM-A2C reward | `GenDAC_lstma2c_reward_function` |
| GAN-DDQN reward | `GenDAC_ganddqn_reward_function` |
| Weighted sum of SE and SSR | `GenDAC_weight_sum` |

After obtaining and exporting the corresponding experimental results as described in Section 6, the reward-function ablation figures can be generated using:

```bash
python Graph_generator_github/reward_function.py
```

The generated figures are automatically stored under:

```text
Test_Figures/reward_function/
```

---
## 9. Configure Sensitivity Analysis

This section describes how to modify `GenDAC.py` to reproduce or extend the sensitivity analyses of two important GenDAC parameters: the reconstruction-loss weight $\lambda$ and the action-logit clipping bound.

The experimental CSV results used for the sensitivity analyses in the thesis are already included under:

```text
Outcome_github/CSVs/
```

The corresponding figure-generation scripts are:

```text
Graph_generator_github/lambda.py
Graph_generator_github/bound.py
```

Therefore, the provided sensitivity-analysis figures can be reproduced directly using the existing CSV results without rerunning the training experiments.

The following subsections first describe which directories under `Outcome_github/CSVs/` correspond to each sensitivity-analysis setting, and then explain the modifications required in `GenDAC.py` when rerunning the experiments with different parameter values.

### 9.1 Sensitivity Analysis of Reconstruction-Loss Weight $\lambda$

The reconstruction-loss weight $\lambda$ controls the contribution of the reconstruction loss during the optimization of the diffusion actor.

The experimental CSV results for the $\lambda$ sensitivity analysis are already provided for random seeds `124`, `125`, `126`, `127`, and `128`.

For each random seed, the following directories correspond to the evaluated $\lambda$ settings:

| CSV Directory | $\lambda$ Setting | Type |
|---|---|---|
| `GenDAC_lam_1_csv/` | $\lambda = 1$ | Constant |
| `GenDAC_lam_05_csv/` | $\lambda = 0.5$ | Constant |
| `GenDAC_lam_0001_csv/` | $\lambda = 0.001$ | Constant |
| `GenDAC_csv/` | $\lambda: 0.5 \rightarrow 0.001$ | Cosine decay |
| `GenDAC_lam_05_00001_csv/` | $\lambda: 0.5 \rightarrow 0.0001$ | Cosine decay |

For example, for seed `124`, the corresponding CSV results are stored under:

```text
Outcome_github/CSVs/seed_124/
├── GenDAC_lam_1_csv/          # λ = 1
├── GenDAC_lam_05_csv/         # λ = 0.5
├── GenDAC_lam_0001_csv/       # λ = 0.001
├── GenDAC_csv/                 # λ: 0.5 → 0.001
└── GenDAC_lam_05_00001_csv/   # λ: 0.5 → 0.0001
```

The same directory names are provided for seeds `125`, `126`, `127`, and `128`.

These directories are the experimental inputs used by:

```text
Graph_generator_github/lambda.py
```

#### Configure $\lambda$ in `GenDAC.py`

In `GenDAC.py`, the initial value of $\lambda$ is specified in **Line 377**:

```python
initial_lambda = 0.5
```

During training, the current value of $\lambda$ is determined in **Lines 672–679**:

```python
current_lambda = get_lambda(
    current_step= frame,
    start_step= batch_size * 3,
    end_step= 6000,
    start_lambda= initial_lambda,
    end_lambda= 0.001
)
# current_lambda = initial_lambda
```

With this configuration, $\lambda$ starts from `0.5` and gradually decreases to `0.001` using cosine decay.

#### Use a Constant $\lambda$

To evaluate a constant $\lambda$, replace **Lines 672–679** with:

```python
current_lambda = initial_lambda
```

Then modify **Line 377** to the desired value.

For example, to evaluate $\lambda = 1$:

```python
initial_lambda = 1.0
```

To evaluate $\lambda = 0.5$:

```python
initial_lambda = 0.5
```

To evaluate $\lambda = 0.001$:

```python
initial_lambda = 0.001
```

#### Use a Dynamically Decaying $\lambda$

To evaluate a dynamically decaying $\lambda$, keep the `get_lambda()` function call in **Lines 672–678**:

```python
current_lambda = get_lambda(
    current_step= frame,
    start_step= batch_size * 3,
    end_step= 6000,
    start_lambda= initial_lambda,
    end_lambda= 0.001
)
```

The initial value is controlled by **Line 377**, while the final value is specified by `end_lambda` in **Line 677**.

For example, to decrease $\lambda$ from `0.5` to `0.001`:

```python
# Line 377
initial_lambda = 0.5
```

```python
# Line 677
end_lambda = 0.001
```

To decrease $\lambda$ from `0.5` to `0.0001`, replace **Line 677** with:

```python
end_lambda = 0.0001
```

The provided sensitivity-analysis figures can be generated directly using:

```bash
python Graph_generator_github/lambda.py
```

The generated figures are automatically stored under:

```text
Test_Figures/lambda/
```

### 9.2 Sensitivity Analysis of Action-Logit Clipping Bound

The action-logit clipping bound controls the allowable range of the action logits generated by GenDAC. For a clipping bound $b$, the action logits are constrained within $[-b,b]$ before being converted into inter-slice resource-allocation proportions.

The experimental CSV results for the clipping-bound sensitivity analysis are already provided for random seeds `124`, `125`, `126`, `127`, and `128`.

For each random seed, the following directories correspond to the evaluated clipping-bound settings:

| CSV Directory | Clipping Bound $b$ | Scaling Factor $\Omega$ |
|---|---:|---:|
| `GenDAC_max_1_csv/` | 1 | 3.0 |
| `GenDAC_max_2_csv/` | 2 | 1.5 |
| `GenDAC_csv/` | 3 | 1.0 |
| `GenDAC_max_4_csv/` | 4 | 1.0 |
| `GenDAC_max_5_csv/` | 5 | 1.0 |

For example, for seed `124`, the corresponding CSV results are stored under:

```text
Outcome_github/CSVs/seed_124/
├── GenDAC_max_1_csv/   # bound = 1, Ω = 3.0
├── GenDAC_max_2_csv/   # bound = 2, Ω = 1.5
├── GenDAC_csv/         # bound = 3, Ω = 1.0
├── GenDAC_max_4_csv/   # bound = 4, Ω = 1.0
└── GenDAC_max_5_csv/   # bound = 5, Ω = 1.0
```

The same directory names are provided for seeds `125`, `126`, `127`, and `128`.

These directories are the experimental inputs used by:

```text
Graph_generator_github/bound.py
```

#### Configure the Clipping Bound in `GenDAC.py`

Three settings in `GenDAC.py` are relevant to this sensitivity analysis.

The clipping bound returned during training is specified in **Line 60**:

```python
return 3
```

The initial clipping bound used to construct the diffusion actor and optimizer is specified in **Line 349**:

```python
initial_max_action = 3
```

The corresponding action-logit scaling factor $\Omega$ is specified in **Line 353**:

```python
action_scale_factor = 1.0
```

When changing the clipping bound, **Line 60 and Line 349 must be set to the same bound value**. The corresponding `action_scale_factor` should also be configured according to the evaluated setting.

The settings used in the provided sensitivity analysis are:

| Clipping Bound $b$ | `initial_max_action` | `action_scale_factor` ($\Omega$) |
|---:|---:|---:|
| 1 | 1 | 3.0 |
| 2 | 2 | 1.5 |
| 3 | 3 | 1.0 |
| 4 | 4 | 1.0 |
| 5 | 5 | 1.0 |

For example, to evaluate $b=1$, configure:

```python
# Line 60
return 1

# Line 349
initial_max_action = 1

# Line 353
action_scale_factor = 3.0
```

To evaluate $b=2$, configure:

```python
# Line 60
return 2

# Line 349
initial_max_action = 2

# Line 353
action_scale_factor = 1.5
```

For the default setting $b=3$, configure:

```python
# Line 60
return 3

# Line 349
initial_max_action = 3

# Line 353
action_scale_factor = 1.0
```

For $b=4$, configure:

```python
# Line 60
return 4

# Line 349
initial_max_action = 4

# Line 353
action_scale_factor = 1.0
```

For $b=5$, configure:

```python
# Line 60
return 5

# Line 349
initial_max_action = 5

# Line 353
action_scale_factor = 1.0
```

The provided sensitivity-analysis figures can be generated directly using:

```bash
python Graph_generator_github/bound.py
```

The generated figures are automatically stored under:

```text
Test_Figures/bound/
```


