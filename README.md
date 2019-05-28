# speedrun-springboard
Boilerplate repository showing how to speedrun from idea to a reproducible experiment setup with Pytorch.

This repository contains code to train a ResNet on Cifar-10. To get started: [install speedrun](https://github.com/inferno-pytorch/speedrun), [fork this repository](https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private) and [hack away](https://pytorch.org/)! Make sure to also install [tensorboardX](https://github.com/lanpa/tensorboardX) while you're at it. 

## Run Example

To run the example code on CPU, do the following. 

```bash
cd speedrun-springboard
mkdir experiments
python train.py experiments/FIRST-0 --inherit templates/BASE-X
```

This will make a new experiment directory `experiments/FIRST-0` (see [speedrun](https://github.com/inferno-pytorch/speedrun) for more on that). 

Now to train on GPU with a different learning rate, you could do the following: 

```bash
python train.py experiments/FIRST-1 --inherit experiments/FIRST-0 --config.device 'cuda:0' --config.optimizer.kwargs.lr 0.01
```
This will make another experiment directory `experiments/FIRST-1`. 

### Happy Hacking!
