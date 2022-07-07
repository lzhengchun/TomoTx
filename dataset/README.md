The synthesized dataset used in the paper/code can be generated using `simu.py`.

In order to save CPU-RAM to host/pre-load data into main memeory, global shuffling was disabled by default (i.e., dataset are pre-sharded for each worker) in the implementation, but local/rank shuffle is on.
