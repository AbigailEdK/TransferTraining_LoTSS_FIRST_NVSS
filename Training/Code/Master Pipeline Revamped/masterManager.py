# region ABOUT
# ===================================================================================================
# > This script handles running five experiments to verify the effect of random augmentations on training data. Each experiment involves splitting the data into training and testing sets, augmenting the training data, and then training a model using the augmented data. The script uses multiprocessing to run each experiment in a separate process, ensuring that memory is cleared between experiments. The results of each experiment are saved in separate directories for later analysis. The script also includes error handling to check for successful completion of each step and to catch any critical errors that may occur during the subprocess execution.

# > Adjusted from manager.py by Dylan Farge.
# ===================================================================================================
# endregion

print("\n\n" + "="*160 + "\n" + "="*160 + "\n\n")

# region IMPORTS
import sys
import os
import multiprocessing
import numpy as np
from pathlib import Path
from datetime import datetime
from masterUtils import check_directory, checkExist
from masterRunModel import runModel
import masterProcess as process
from masterDataAugmentation import augmentData, splitData, copySplitData
from masterSearchSummary import summarise_stage, save_summary
# endregion

# region PATHS
from paths import RESULTSDIRECTORY, SOURCEDATADIRECTORY
# endregion

# region BOOLS
from bools import TRAINING
# endregion

# region SEARCH SETTINGS
# > One run goes through all three stages in order. Each stage picks its configs from the previous stage's validation
# > results (mean over folds of the lowest validation loss), so the test set is only used in the FINAL stage.
# |  COARSE: decade grid on 5 representative surveys, 2 folds, no transfer learning, no testing.
# |  FINE:   +-half a decade around each survey's COARSE best, all 15 surveys, 5 folds, no transfer learning, no testing.
# |  FINAL:  the FINE best per survey, all 15 surveys, 5 folds, then transfer learning and testing.
STAGES = ["COARSE", "FINE", "FINAL"] # | Remove stages from the front to start later (their results must already exist)
EPOCH_CEILING = 100
EXPERIMENT_NUMBER = 0

ALL_SURVEYS = ["FIRST", "NVSS", "LoTSS", "FL", "LF", "FN", "NF", "LN", "NL", "FLN", "FNL", "NLF", "NFL", "LNF", "LFN"]
SEARCH_SURVEYS = ["FIRST", "NVSS", "LoTSS", "FL", "FN", "LN", "FLN"] # | One per survey combination: each stands in for its channel-order permutations

# > COARSE: whole decades only
COARSE_LEARNING_RATE_RANGE = [1e-5, 1e-4, 1e-3, 1e-2]
COARSE_REGULARISATION_RANGE = [0.0, 1e-4, 1e-3, 1e-2] # | L2 coefficient on the Dense(500) layer
COARSE_WEIGHT_DECAY_RANGE = [1e-4, 1e-2]

# > Optional manual overrides as (lr, reg, wd). Leave as None to take the value from the previous stage's results.
FINE_CENTRES = {survey: None for survey in SEARCH_SURVEYS}
BEST_PARAMETERS = {survey: None for survey in ALL_SURVEYS}

STAGE_SETTINGS = { # | (surveys, max_folds, run_transfer, run_testing)
    "COARSE": (SEARCH_SURVEYS, 2, False, False),
    "FINE": (ALL_SURVEYS, 5, False, False),
    "FINAL": (ALL_SURVEYS, 5, True, True),
}
# endregion

# region SEARCH HELPERS
def representative_survey(survey):
    '''
    Returns the survey whose COARSE result is used as the FINE centre: the search survey with the same set of surveys, in any channel order (e.g. LF uses FL, NF uses FN, NL uses LN, every trio uses FLN).
    '''
    for representative in SEARCH_SURVEYS:
        if len(survey) > 1 and len(representative) == len(survey) and set(representative) == set(survey):
            return representative
    return survey

def fine_grid(lr, reg, wd):
    '''
    Returns the (lr, reg, wd) configs within half a decade of a COARSE best: 3 learning rates x 3 regularisations x 3 weight decays.
    A regularisation of 0 is searched as 0 and two small values, since there is nothing below it.
    '''
    half_decade = np.sqrt(10)
    lrs = [lr / half_decade, lr, lr * half_decade]
    regs = [0.0, 3e-5, 1e-4] if reg == 0 else [reg / half_decade, reg, reg * half_decade]
    wds = [wd / half_decade, wd, wd * half_decade]
    return [(l, r, w) for l in lrs for r in regs for w in wds]
# endregion

# region STAGES
def best_parameters(stage):
    '''
    Returns {survey: (lr, reg, wd)} with the lowest mean validation loss in the given stage, and saves that stage's full ranking.
    '''
    summary = summarise_stage(stage, EXPERIMENT_NUMBER)
    save_summary(summary, stage, EXPERIMENT_NUMBER)
    best = summary[summary["rank"] == 1]
    return {row["survey"]: (row["lr"], row["reg"], row["wd"]) for _, row in best.iterrows()}

def build_configs(stage):
    '''
    Returns {survey: [(lr, reg, wd), ...]} for the given stage.
    '''
    surveys = STAGE_SETTINGS[stage][0]

    if stage == "COARSE":
        grid = [(lr, reg, wd) for lr in COARSE_LEARNING_RATE_RANGE for reg in COARSE_REGULARISATION_RANGE for wd in COARSE_WEIGHT_DECAY_RANGE]
        return {survey: grid for survey in surveys}

    if stage == "FINE":
        centres = best_parameters("COARSE")
        centres.update({survey: value for survey, value in FINE_CENTRES.items() if value is not None})
        configs = {}
        for survey in surveys:
            representative = representative_survey(survey)
            if representative not in centres:
                raise ValueError(f"No COARSE result for {representative}, which {survey} needs as its FINE centre.")
            print(f"FINE centre for {survey} (from {representative}): lr={centres[representative][0]:.3g}, reg={centres[representative][1]:.3g}, wd={centres[representative][2]:.3g}")
            configs[survey] = fine_grid(*centres[representative])
        return configs

    if stage == "FINAL":
        best = best_parameters("FINE")
        best.update({survey: value for survey, value in BEST_PARAMETERS.items() if value is not None})
        configs = {}
        for survey in surveys:
            if survey not in best:
                raise ValueError(f"No FINE result for {survey}.")
            print(f"FINAL parameters for {survey}: lr={best[survey][0]:.3g}, reg={best[survey][1]:.3g}, wd={best[survey][2]:.3g}")
            configs[survey] = [best[survey]]
        return configs

    raise ValueError(f"Unknown stage: {stage}")

def run_in_process(args, description):
    '''
    Runs masterRunModel.runModel in a fresh process. Each process exits when its run is done, so all GPU memory is
    released between runs instead of building up until TensorFlow runs out.
    '''
    p = multiprocessing.Process(target=runModel, args=args)
    p.start()
    p.join()

    # > Check if the process crashed
    if p.exitcode != 0:
        print(f"CRITICAL: {description} crashed with exit code {p.exitcode}. Rerun the script to resume: finished runs are skipped.")
        sys.exit(p.exitcode)

def run_stage(stage):
    surveys, max_folds, run_transfer, run_testing = STAGE_SETTINGS[stage]
    configs = build_configs(stage)

    experiments = []
    for survey in surveys:
        for lr, reg, wd in configs[survey]:
            experiments.append((EPOCH_CEILING, survey, float(lr), float(reg), float(wd)))

    print("~"*160 + "\n")
    print(f">>>>> Starting stage {stage} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<<< \n")
    print(f"Surveys: {len(surveys)} | configs: {len(experiments)} | folds: {max_folds} | model fits before transfer learning: {len(experiments) * max_folds} | transfer learning: {run_transfer} | testing: {run_testing}")
    print("~"*160 + "\n")

    training_dir = stage_directory(stage) / "Training"

    # > Training: one process per config. Finished configs are skipped here, before starting a process for them.
    for i, experiment in enumerate(experiments):
        epoch_ceiling, survey, lr, reg, wd = experiment
        folder = process.folder_construct(base_model="NONE", model=survey, surveys=survey, lr=lr, base_reg=reg, reg=reg, wd=wd, epoch_ceiling=epoch_ceiling, save_dir=str(training_dir))
        if os.path.exists(os.path.join(folder, "best_epoch.npy")):
            print(f"{stage} config {i+1}/{len(experiments)} already trained, skipping.")
            continue
        print(f"\n>>>>> {stage} config {i+1}/{len(experiments)} <<<<<")
        run_in_process(([experiment], EXPERIMENT_NUMBER, stage, max_folds, False, False), f"{stage} config {i+1}/{len(experiments)}")

    # > Transfer learning: one process per base model
    if run_transfer:
        base_folders = sorted(folder for folder in training_dir.iterdir() if folder.is_dir() and folder.name.startswith("baseNONE_"))
        for i, folder in enumerate(base_folders):
            print(f"\n>>>>> {stage} transfer learning from base {i+1}/{len(base_folders)}: {folder.name} <<<<<")
            run_in_process(([], EXPERIMENT_NUMBER, stage, max_folds, True, False, str(folder)), f"Transfer learning from {folder.name}")

    # > Testing: one process per trained model (scratch and transfer)
    if run_testing:
        model_folders = sorted(folder for folder in training_dir.iterdir() if folder.is_dir())
        for i, folder in enumerate(model_folders):
            print(f"\n>>>>> {stage} testing model {i+1}/{len(model_folders)}: {folder.name} <<<<<")
            run_in_process(([], EXPERIMENT_NUMBER, stage, max_folds, False, True, str(folder)), f"Testing {folder.name}")

    print("~"*160 + "\n")
    print(f">>>>> Finished stage {stage} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<<< \n")
    print("~"*160 + "\n")

def stage_directory(stage):
    '''
    The folder a stage saves to: Search_<stage> for the searches, the experiment folder itself for FINAL (same as masterRunModel.runModel).
    '''
    experiment_directory = Path(RESULTSDIRECTORY) / f"Experiment_{EXPERIMENT_NUMBER}"
    return experiment_directory if stage == "FINAL" else experiment_directory / f"Search_{stage}"
# endregion

# region MAIN
def main():
    # > Keep existing results and data, so a later stage never wipes an earlier one. Delete folders by hand to start over.
    if TRAINING: check_directory(RESULTSDIRECTORY, replace=False)
    if TRAINING: check_directory(SOURCEDATADIRECTORY, replace=False)

    start_time = datetime.now()
    print(f"Starting Master Manager at {start_time.strftime('%Y-%m-%d %H:%M:%S')} \n\n\n")

    # === Prepare Data ===
    sourceDataExist = checkExist(SOURCEDATADIRECTORY, num_files=6)
    if TRAINING:
        if sourceDataExist == False:
            splitData(SOURCEDATADIRECTORY) # | Split data into training and testing sets, and do not repeat for every experiment.
        else: print(f"Source data already exists in {SOURCEDATADIRECTORY}, proceeding with training.")

    EXPERIMENTDIRECTORY = os.path.join(RESULTSDIRECTORY, f"Experiment_{EXPERIMENT_NUMBER}")

    augmentDataExist = checkExist(os.path.join(EXPERIMENTDIRECTORY, "Data"), num_files=7)
    if TRAINING:
        if augmentDataExist == False:
            copySplitData(SOURCEDATADIRECTORY, EXPERIMENTDIRECTORY) # Copy source data into experiment folder
            augmentData(os.path.join(EXPERIMENTDIRECTORY, "Data")) # Augment the data in the experiment folder
        else: print(f"Augmented data already exists in {os.path.join(EXPERIMENTDIRECTORY, 'Data')}, proceeding with training.")

    # > Check that the augmentation process completed successfully by verifying the existence of the expected output files.
    required_file = os.path.join(EXPERIMENTDIRECTORY, "Data", "X_train_val.npy")
    if not os.path.exists(required_file):
        print(f"ERROR: Augmentation failed! {required_file} not found.")
        sys.exit(1)

    # === Run Stages ===
    for stage in STAGES:
        run_stage(stage)

    print(f"All stages finished in {datetime.now() - start_time}.")
    print("=" * 160 + "\n" + "=" * 160 + "\n\n")
# endregion

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
