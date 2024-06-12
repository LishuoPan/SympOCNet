# SympOCNet Daily Progres

## Before 2024-06-10

- Ran `example_maze.py` with all default settings and saved to baseline

## 2024-06-10

- Added the `visualize_losses.py` script to visualize `original_losses.log`
- Rerun `example_maze.py` with all default settings and saved to baseline_rerun
  - Losses increased to ~25k iters and then decrease to about 40 at the end of 10k iters
  - PINN loss (loss_sympnet) dominated after 40k iters
  - Very unstable loss_aug_lag and loss_aug_bd
- Added parametric NN with params [here](https://github.com/LishuoPan/SympOCNet/blob/8411d8dcc23d2ca3d0f60f8b90acdf6cb133b6ce/learner_zhen/nn.py#L21)
  - Saved results to `parameters_nn_added`, best model at 45k iters
  - For now, loss_aug_bd completed exploded -> NEED TO DEBUG

## 2024-06-11

- Ran `main.py` with parameters_nn but without loss_aug_bd, also have the `plot_simple` to visualize results every 1k iters, results stored in 2024-06-10_23-54-54
  - According to the predicted trajectories, doesn't seem to have a lot of change
  - Losses are reasonable, but loss_bd are dominating over PINN loss

- Run original experiment again without loss_aug_bd, results stored in 2024-06-11_11-46-50
  - Was not meant to run without loss_aug_bd
  - Trajectories quite different -> maybe the environment setup allows different trajectories with low costs
  - Loss lowered to about 100 at the end, best model at 70k iters

- Original experiment on demo_32 setup, results stored in 2024-06-11_13-46-44
  - The trajectories look pretty reasonable, though the plots are clipped
  - loss_aug_bd exploded -> seems that the exploding loss isn't a big problem
  - Best model at iter 20k -> very early on

Need to perform collision check and higher order derivatives check on the outputs

## TODO:

- [ ] Figure out what cost actually is 
- [ ] Make sure LBGFS works
- [ ] Spectral norm
- [ ] 3D experiment
- [ ] 32 with parameters_nn