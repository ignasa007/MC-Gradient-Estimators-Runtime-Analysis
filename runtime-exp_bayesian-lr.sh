#!/bin/bash
set -e

for N in `seq 30 5 34`; do
    python -m monte_carlo_gradients.main "score_function" "none"       $N # &&
    # python -m monte_carlo_gradients.main "pathwise"       "none"       $N &&
    # python -m monte_carlo_gradients.main "measure_valued" "none"       $N &&
    # python -m monte_carlo_gradients.main "score_function" "delta"      $N &&
    # python -m monte_carlo_gradients.main "pathwise"       "delta"      $N &&
    # python -m monte_carlo_gradients.main "score_function" "moving_avg" $N;
done