# Monte Carlo Gradient Estimation in Machine Learning

This is the example code for the following paper, adapted for presentation for the course Machine Learning Seminar (COMP0168) at UCL:

> Shakir Mohamed, Mihaela Rosca, Michael Figurnov, Andriy Mnih
  *Monte Carlo Gradient Estimation in Machine Learning*.  [\[arXiv\]](https://arxiv.org/abs/1906.10652).

## Running the code

The code contains:

  * the implementation the score function, pathwise and measure valued estimators `gradient_estimators.py`.
  * the implementation of control variates `control_variates.py`.
  * a `main.py` file, edited to compute the computational cost per iteration of the Bayesian Logistic regression experiments in the paper.
  * a `config.py` file used to configure experiments.

To install the required dependencies:

```
  source monte_carlo_gradients/setup.sh
```

A runtime analysis of the experiments from Section 3 can be found in `runtime-exp_dummy-data.ipynb`. To analyse the runtime for the Bayesian LR experimtents, run

```
  python -m monte_carlo_gradients.main ${estimator} ${control-variate} ${N}
```
where `estimator` is one of ("score_function", "pathwise", "measure_valued"), `control-variate` is one of ("none", "moving_avg", "delta") and `N` is an positive integer. This saves the results in the directory `./results/N-${N}/estimator-${estimator}/cv-${control-variate}`.

Note that no control-variate can be used with measure-valued gradient because the weak derivative decomposition will have to be analytically computed (which is a pain). 

To save results for all methods with a particular sequence of values of `N`, edit the iterator in `runtime-exp_bayesian-lr.sh` appropriately and run

```
  source runtime-exp_bayesian-lr.sh
```

The current implementation runs from `N = 30` to `N = 400` with a step size of `5`. Finally, the results can be compared in `analysis.ipynb`.

## Disclaimer

This code has largely been borrowed from the [original work](https://github.com/google-deepmind/mc_gradients/tree/master), and a few changes have been made to focus solely on the runtime computation. For any other experiments, we suggest referring to the [original code base](https://github.com/google-deepmind/mc_gradients/tree/master).