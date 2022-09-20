# MuZero for Jass
MuZero is a model based reinforcement learning method using deep learning and therefore requires training.
The data is generated either through self-play or reanalysing existing data (tfrecord format as described in [jass-ml-py](https://github.com/thomas-koller/jass-ml-py/tree/master/jass/features) repo.

## Setup
Install the package

```bash
$ pip install -v -e .
```

Run tests to verify local setup

```bash
$ sjmz (--nodocker) --test
```

And finally start the container hosting the baselines with

```bash
$ sjmz --baselines
```

## Training
The MuZero training process is implemented in a distributed manner.
The docker-compose service `trainer` is the master container which will gather all the data, train the networks
and evaluate them asynchronously on different metrics.
In the folder `resources/data_collectors` there are different compose files for different machines to host data collectors.
They all assume that the master container is running on the `ws03` machine (IP: 10.180.39.13).
If this would not be the case, the IP in the files must be adapted.
To start the training process first run 

```bash
$ sjmz --attach train --file experiments/experiment-1.json
```

and wait until the flask server started hosting. Then start the data collectors on the respective machines


```bash
$ sjmz collect --machine (gpu03|e01|...)
```

available configurations are stored at `resources/data_collectors`. The collectors should then register them on the master container and start to collect data.
Once the replay buffer has been filled, the optimization procedure will start and the corresponding metrics will
be logged to wandb.ai at the configured location.


#Evaluate
To evaluate run
```bash
$ sjmz eval --files experiment-1/dmcts.json
```
