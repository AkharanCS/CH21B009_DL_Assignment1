# DA6401 - Assignment 1

## Directories
- **sample_runs**: contains a sample run of the network using every optimizer. If you wish to run these files, please bring them out to the root directory as it imports classes from the root directory.

## Files 
- **`config.yaml`**: contains the hyperparameter space used for running wandb sweeps
- **`NN.py`**: contains all the classes and functions constituting the feedforward neural network. (solution for Q2)
- **`GD.py`**: contains all the classes and functions constituting every type of optimizer. (solution for Q3)
- **`utils.py`**: contains one-hot encoding function to encode the classification labels.

All the other files import the above three files to do the required tasks.

- **`Q4.py`**: contains code for performing the wandb sweep required in Q4
- **`Q7.py`**: contains code required for Q7
- **`Q8.py`**: contains code for comparing squared-error and cross-entropy loss as required in Q8
- **`Q10.py`**: contains code for running the best network configurations on mnist as required in Q10
- **`train.py`**: contains code which accepts all the commands given under code specification. 

## How to run (follow this order)
1. Clone the repository:
   ```bash
   git clone https://github.com/AkharanCS/CH21B009_DL_Assignment1.git
   ```
2. Run `NN.py`. <br>
3. Run `GD.py`. <br>
4. Run `utils.py`. <br>
5. Save `config.yaml` file. <br>
6. All the other files,(`Q1.py`,`Q4.py`,`Q7.py`,`Q8.py`,`Q10.py`) can be run in any order. <br>
7. **`train.py`** can be run via the command line using the following command:
    ```bash
    python train.py --wandb_entity your_account_name --wandb_project your_project_name
    ```
8. If needed, to run files inside `sample_runs`, please move them to the root directory. <br>


