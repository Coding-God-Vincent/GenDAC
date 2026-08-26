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

Since the configuration procedure is the same for all algorithms, `GenDAC.py` is used as an example in this section.

### 4.1 Configure Experiment Names and Random Seeds

In `GenDAC.py`, replace **Lines 285–286** with the desired experiment names and random seeds.

For example, to run three independent experiments, replace Lines 285–286 with:

```python
exps = ['exp1', 'exp2', 'exp3']
seeds = [124, 125, 126]
```

Each experiment name in `exps` corresponds to the random seed at the same position in `seeds`:

```text
exp1 → seed 124
exp2 → seed 125
exp3 → seed 126
```

To run only one experiment, replace Lines 285–286 with:

```python
exps = ['exp1']
seeds = [124]
```

Make sure that `exps` and `seeds` have the same number of entries.

For the baseline algorithms, modify the corresponding lines shown below:

| Algorithm | Experiment Name | Random Seed |
|---|---:|---:|
| `GenDAC.py` | Line 285 | Line 286 |
| `GANDDQN.py` | Line 55 | Line 51 |
| `LSTM_A2C.py` | Line 28 | Line 24 |
| `PPO.py` | Line 23 | Line 24 |
| `SAC.py` | Line 29 | Line 31 |
| `Hard_Slicing.py` | Line 14 | Line 15 |


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

For example:

```text
Logs_github/GenDAC/exp1/tensorboard/
Logs_github/GenDAC/exp2/tensorboard/
Logs_github/GenDAC/exp3/tensorboard/
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
    └── exp3/
        ├── QoE.png
        ├── SE.png
        └── Utility.png
```

The required directories under `Temp_Figures/` are created automatically during execution if they do not already exist. Therefore, there is no need to manually create the output directories before running the training scripts.

These PNG files are intended for quickly inspecting the training behavior of each individual run. For more detailed analysis and figure generation across multiple seeds, the TensorBoard logs can be exported to CSV files as described in Section 6 and processed using the scripts in `Graph_generator_github/` as described in Section 7.

## 5. Real-Time Training Monitoring with TensorBoard

TensorBoard can be used to monitor the training progress in real time. The required TensorBoard package is already included in `requirements.txt`.

### 5.1 Start Training

For example, start training GenDAC in one terminal:

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

Next, configure the random seed and the experiment name by replacing **Lines 90–91**.

For example:

```python
seeds = [124]
algo_name = "Test_GenDAC"
```

The value of `algo_name` is used as the name of the output CSV directory.

The generated CSV files follow the directory structure:

```text
Outcome_github/CSVs_new/seed_<seed>/<algo_name>_csv/
```

For example, with:

```python
seeds = [124]
algo_name = "Test_GenDAC"
```

the generated CSV files are stored under:

```text
Outcome_github/CSVs_new/seed_124/Test_GenDAC_csv/
```

The output directories are created automatically if they do not already exist. Therefore, `CSVs_new/`, `seed_<seed>/`, and `<algo_name>_csv/` do not need to be created manually.

If multiple experiments are exported at the same time, replace **Lines 90–91** accordingly.

For example:

```python
seeds = [124, 125, 126]
algo_name = "Test_GenDAC"
```

The number and order of entries in `seeds` must be consistent with the TensorBoard event file paths specified in `event_path_moving`, as described in Section 6.2.

### 6.2 Specify the TensorBoard Event File

The TensorBoard event files to be exported are specified using `event_path_moving`.

In `CSV_generator_github/csv_generator1.py`, replace **Lines 94–97** with the path of the TensorBoard event file generated by the corresponding training experiment.

TensorBoard event files generated during training can be found under:

```text
Logs_github/<algorithm>/<experiment>/tensorboard/
```

For example:

```text
Logs_github/GenDAC/exp1/tensorboard/
```

Inside the corresponding `tensorboard/` directory, locate the TensorBoard event file whose name starts with:

```text
events.out.tfevents
```

Copy the path of this event file and assign it to `event_path_moving`. The TensorBoard event file itself does not need to be moved, copied, or replaced. (Both **relative paths** and **absolute paths** can be used.)

For example, replace **Lines 94–97** with the following when using a relative path from the project root directory:

```python
event_path_moving = [
    'Logs_github/GenDAC/exp1/tensorboard/events.out.tfevents...'
]
```

Alternatively, an absolute path can be used:

```python
event_path_moving = [
    '/home/user/GenDAC/Logs_github/GenDAC/exp1/tensorboard/events.out.tfevents...'
]
```

When using a relative path, the path should be relative to the directory from which the CSV generator is executed. In the examples in this README, `csv_generator1.py` is executed from the project root directory.

If multiple seeds or experiments are exported at the same time, specify the corresponding TensorBoard event file paths in `event_path_moving` in the same order as the entries in `seeds`.

For example, replace **Lines 90–91** with:

```python
seeds = [124, 125, 126]
algo_name = "Test_GenDAC"
```

and replace **Lines 94–97** with:

```python
event_path_moving = [
    'Logs_github/GenDAC/exp1/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp2/tensorboard/events.out.tfevents...',
    'Logs_github/GenDAC/exp3/tensorboard/events.out.tfevents...'
]
```

In this example:

```text
seed 124 → exp1 TensorBoard event file
seed 125 → exp2 TensorBoard event file
seed 126 → exp3 TensorBoard event file
```

Therefore, the number of entries in `seeds` and `event_path_moving` must be the same, and their order must correspond to each other.

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

## 7. Generate Figures from CSV Results

The scripts in `Graph_generator_github/` can be used to read experimental CSV files and generate evaluation figures.

Each script corresponds to a specific experiment, comparison, or ablation study.

For example:

```text
Graph_generator_github/6_algos.py
```

generates the main comparison figures for GenDAC, GAN-DDQN, LSTM-A2C, Hard Slicing, PPO, and SAC.

### 7.1 Configure the CSV Input

The figure-generation scripts specify the algorithm names, random seeds, CSV paths, and other experiment-specific parameters.

This section uses `Graph_generator_github/6_algos.py` as an example.

#### Configure the Algorithm Names

In `Graph_generator_github/6_algos.py`, **Lines 41–46** specify the names of the algorithms whose CSV directories will be loaded:

```python
algo_name1 = "GenDAC"
algo_name2 = "GANDDQN"
algo_name3 = "LSTM_A2C"
algo_name4 = "Hard_Slicing"
algo_name5 = "PPO"
algo_name6 = "SAC"
```

These names must match the corresponding `<algo_name>_csv` directory names.

For example, if the CSV files of GenDAC are stored under:

```text
Outcome_github/CSVs_new/seed_124/Test_GenDAC_csv/
```

replace **Line 41** with:

```python
algo_name1 = "Test_GenDAC"
```

The remaining algorithm names in **Lines 42–46** should be modified in the same way when necessary.

The legend labels displayed in the generated figures are defined in **Line 49**:

```python
labels = ["GenDAC", "GAN-DDQN", "LSTM-A2C", "Hard Slicing", "PPO", "SAC"]
```

Modify this line only if different names should be displayed in the figure legends.

#### Configure the Random Seeds

The random seeds used to load the experimental results are specified in **Line 51**:

```python
seeds = [124, 125, 126, 127, 128]
```

Replace **Line 51** with the seeds corresponding to the CSV results to be processed.

For example, to generate figures using only seed `124`:

```python
seeds = [124]
```

The specified seeds must correspond to the available directories:

```text
Outcome_github/CSVs/seed_<seed>/
```

or:

```text
Outcome_github/CSVs_new/seed_<seed>/
```

depending on the configured CSV input path.

#### Configure the CSV Input Path

The current `6_algos.py` is configured to read the experimental CSV files used to generate the figures reported in the thesis from:

```text
Outcome_github/CSVs/
```

The CSV input path is specified separately for Utility, SE, and SSR. In `6_algos.py`, the corresponding paths are located at:

```text
Line 74  → System Utility
Line 125 → Spectral Efficiency (SE)
Line 186 → Slice Satisfaction Rate (SSR)
```

The current setting at each of these lines is:

```python
csv_path = Path("Outcome_github/CSVs") / f"seed_{seeds[j]}"
```

The CSV files stored in `Outcome_github/CSVs/` are the experimental results used to generate the figures reported in the thesis.

If new experiments are executed and their TensorBoard results are exported using `CSV_generator_github/csv_generator1.py`, the newly generated CSV files are stored under:

```text
Outcome_github/CSVs_new/
```

To generate figures using these newly generated experimental results, replace **Lines 74, 125, and 186** with:

```python
csv_path = Path("Outcome_github/CSVs_new") / f"seed_{seeds[j]}"
```

All three lines must be modified so that Utility, SE, and SSR are loaded from the same CSV directory.

For example, suppose the CSV generator is configured as:

```python
seeds = [124]
algo_name = "Test_GenDAC"
```

and the generated files are stored under:

```text
Outcome_github/CSVs_new/seed_124/Test_GenDAC_csv/
```

Then configure `6_algos.py` as follows:

Replace **Line 41** with:

```python
algo_name1 = "Test_GenDAC"
```

Replace **Line 51** with:

```python
seeds = [124]
```

Replace **Lines 74, 125, and 186** with:

```python
csv_path = Path("Outcome_github/CSVs_new") / f"seed_{seeds[j]}"
```

The algorithm names, seeds, and CSV input paths must be consistent with the directories generated by `CSV_generator_github/csv_generator1.py`.

> **Note:** The algorithm names, random seeds, CSV paths and other experiment-specific settings currently defined in `Graph_generator_github/` are configured for the experimental results used in the thesis. When generating figures from newly obtained experimental results, modify these settings according to the corresponding experiment.

### 7.2 Configure the Figure Output

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

### 7.3 Generate the Figures

After configuring the CSV input path, algorithm names, random seeds, and other experiment-specific settings, run the desired figure-generation script from the project root directory.

For example, to generate the main six-algorithm comparison figures:

```bash
python Graph_generator_github/6_algos.py
```

The script reads the configured CSV files, processes the experimental results, and generates the corresponding evaluation figures.

For the default configuration of `6_algos.py`, the generated figures are stored under:

```text
Test_Figures/6_algos/
```

Other figure-generation scripts can be executed in the same way.

For example:

```bash
python Graph_generator_github/nr_oriented.py
python Graph_generator_github/throughput.py
python Graph_generator_github/reward_function.py
```

Before running a figure-generation script with newly generated experimental data, make sure that its CSV input path, algorithm names, random seeds, and other experiment-specific settings are consistent with the CSV files generated in Section 6.

