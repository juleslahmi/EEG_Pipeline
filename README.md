EEG_ML

EEG classification pipeline (Control vs Dyslexic) with LOSO/group-k-fold training, evaluation, and result aggregation.

Setup

1. Create/activate a Python environment.
2. Install dependencies:

	pip install -r requirements.txt

Run

- One Experiment: `run_experiment.sh`
- Evaluate across seeds: `run_seeds.sh`

In any of the shell file, all parameters are customisable to obtain any of the desired results/

Data is expected under `Data/Control` and `Data/Dyslexic`.

