#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

clear 
echo "DreamerV3"
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    max_iters=50 \
    ++experiment.name=q3_dreamer

echo "Final Training"
python dreamer_model_trainer.py \
    model_type=dreamer \
    planner.type=policy_guided_cem \
    planner.horizon=10 \
    planner.num_samples=100 \
    planner.num_elites=10 \
    max_iters=100 \
    ++load_policy=outputs/2026-03-02/q4_dreamer_policy_cem/model_epoch_41_batch_15.pth \
    ++load_world_model=outputs/2026-03-01/q3_dreamer_cem/model_epoch_41_batch_15.pth \
    ++experiment.name=final_dreamer_policy_cem