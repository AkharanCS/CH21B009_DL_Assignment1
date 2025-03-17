# DA6401 - Assignment 1

## Directories
- **sample_runs**: contains a sample run of the network using every optimizer. If you wish to run these files, please bring them out to the root directory as it imports classes from the root directory.

## Files 
- **config.yaml**: contains the hyperparameter space used for running wandb sweeps
- **`NN.py`**: contains all the classes and functions constituting the feedforward neural network. (solution for Q2)
- **`GD.py`**: contains all the classes and functions constituting every type of optimizer. (solution for Q3)
- **`utils.py`**: contains one-hot encoding function to encode the classification labels.

All the other files import these three files to do the required tasks.

- **`Q4.py`**: contains code for performing the wandb sweep required in Q4
- **`Q7.py`**: contains code required for Q7
- **`Q8.py`**: contains code for comparing squared-error and cross-entropy loss as required in Q8
- **`Q10.py`**: contains code for running the best network configurations on mnist as required in Q10
- **`train.py`**: contains code which accepts all the commands given under code specification. (one-stop file to check everything) 

## How to run
1. Clone this repository <br>
2. If required, change the port numbers for both the containers in the `docker-compose.yaml`. <br>
3. I have set it up to be 7100:5432 for postgres container and 8089:8089 for the application. <br>
4. You can change the container names in the `docker-compose.yaml` file if required. <br> 
5. You do not have to change any details regarding the database as a new container for the database is being created from scratch. <br>
6. Only run the follwing command `docker-compose up --build` <br>

## To view the database in pgAdmin
Add a new server connection using the following credentials: <br>
- connection name : anything <br>
- hostname : localhost <br>
- port : 7100 <br> 
- maintenance database : assignment4-db <br>
- username : admin <br>
- password : assignment4 <br>
These credentials are directly based on the environment variables from the Dockerfile inside task1-db. Port number is from the `docker-compose.yaml` file. <br>

