# SympOCNet Daily Progres


## Before 2024-06-10

- Ran `example_maze.py` with all default settings and saved to baseline

## 2024-06-10

- Rerun `example_maze.py` with all default settings and saved to baseline_rerun
- Added the `visualize_losses.py` script to visualize `original_losses.log`
  - Losses increased to ~25k iters and then decrease to about 40 at the end of 10k iters
  - PINN loss (loss_sympnet) dominated after 40k iters
  - Very unstable loss_aug_lag and loss_aug_bd

- Added parametric NN with params [here](https://github.com/LishuoPan/SympOCNet/blob/8411d8dcc23d2ca3d0f60f8b90acdf6cb133b6ce/learner_zhen/nn.py#L21)
- Saved results to `parameters_nn_added`, best model at 45k iters
- For now, loss_aug_bd completed exploded -> NEED TO DEBUG


TODO:

- [ ] Rewrite `plot_heat`