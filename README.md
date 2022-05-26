# Source code for "From Shapley back to Pearson: Hypothesis Testing Via the Shapley Value"

This is the source code for the paper "From Shapley back to Pearson: Hypothesis Testing Via the Shapley Value". The source code will be made publicly available in a non-anonymized repository on GitHub after the review process.

## Instructions to reproduce results

Here, we describe how to reproduce the results presented in the paper.

0. **Install `requirements.txt`**. The source code was developed using a Python `3.8.13` environment. If possible, we recommend installing the required packages listed in the `requirements.txt` file in a virtual environment running the same version of Python to reduce the risk of incompatibilities.

1. **Install `hshap`**. The `./hshap/` folder contains the modified source code of the original `hshap` package, publicly available at [https://github.com/Sulam-Group/h-shap](https://github.com/Sulam-Group/h-shap). We extend the original `./hshap/hshap/src.py` file to return the `p`-values of the SHAPLIT tests in every Shapley coefficient. Changes to the original source code are highlighted with tags, for example:

    ```python
    [...]
    while nodes.shape[1] < stop_l:
    scores = scores.unsqueeze_(1).repeat((1, self.gamma))
    ### START CHANGES ###
    if return_shaplit:
        p = torch.zeros_like(scores)
        p = p.unsqueeze_(2).repeat((1, 1, 2 ** (self.gamma - 1)))
    #### END CHANGES ####
    [...]
    ```

    To install `hshap`, run the following command:

    ```bash
    python -m pip install -e hshap
    ```


2. **Known Boolean function.** The `./experiments/boolean_function.ipynb` notebook contains the code to reproduce the results in Sec. 4.1 of the paper. To reproduce Fig. 1 in the paper, simply run the notebook. Figures should be saved in the `./figures/boolean/` folder.

3. **Synthetic image dataset.** The `./experiments/crosses/` folder contains the code to reproduce the results in Sec. 4.2. of the paper. To reproduce Fig. 2, first run `./experiments/crosses/power_m.py` and `./experiments/crosses/power_sigma.py`, and then run the notebook `./experiments/crosses/figures.ipynb`. Figures should be saved in the `./figures/crosses/` folder.

4. **Real image dataset**. The `./experiments/BBBC041/` folder contains the code to reproduce the results in Sec. 4.3. For ease of reproducibility, `./experiments/BBBC041/pretrained_model` contains the pretrained model used in the paper, and `./experiments/BCC041/train.py` contain the code used to train the model. To reproduce Fig. 3 using the pretrained model included in this archive, run `./experiments/BBBC041/explain.py` to generate explanations for the true positive test images. Then, run the notebook `./experiments/BBBC041/figures.ipynb`. Figures should be saved in the `./figures/BBBC041/` folder.