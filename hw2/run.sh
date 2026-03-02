#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Part 2: Policy-Guided Planning..."
clear
python dreamer_model_trainer.py model_type=simple planner.type=policy_guided_cem planner.horizon=10 planner.num_samples=50 planner.num_elites=5 ++load_policy=outputs/2026-02-28/q2_policy_training/model_epoch_41_batch_15.pth ++experiment.name=q2_policy_training_from_trained
clear

echo "Starting Part 4: Image-Based Policy with DreamerV3..."

python dreamer_model_trainer.py model_type=dreamer planner.type=policy_guided_cem planner.horizon=10 planner.num_samples=50 ++load_policy=outputs/2026-03-01/q4_dreamer_policy/model_epoch_41_batch_15.pth ++load_world_model=outputs/2026-03-01/q3_dreamer_cem/model_epoch_41_batch_15.pth ++experiment.name=q4_dreamer_policy_cem

echo "All tasks completed successfully! Enjoy your time in Portugal!"