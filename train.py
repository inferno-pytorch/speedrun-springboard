import torch
import torchvision
import torchvision.transforms as transforms

from speedrun import BaseExperiment, TensorboardMixin, IOMixin

import models


class CifarTrainer(BaseExperiment, TensorboardMixin, IOMixin):
    # With this, we tell speedrun that the default function to dispatch (via `run`) is `train`.
    DEFAULT_DISPATCH = 'train'

    def __init__(self):
        super(CifarTrainer, self).__init__()
        # The magic happens here.
        self.auto_setup()
        # Build the module
        self._build()

    def _build(self):
        # Build the data loaders
        self._build_loaders()
        # Build model, optimizer, scheduler and criterion.
        # You may use `get` methods to read from the yaml config file of your experiment.
        # Simply use `/` to go down in the yaml hierarchy. The second argument is the default
        # value. `get` can also return dictionaries (if you don't address all the way in), which
        # is useful in combination with the `**kwargs` pattern. Finally, you may use getattr.
        self.model = getattr(models, self.get('model/name', 'ResNet18')) \
            (**self.get('model/kwargs', {})).to(self.device)
        self.optimizer = getattr(torch.optim, self.get('optimizer/name', 'SGD')) \
            (self.model.parameters(),
             **self.get('optimizer/kwargs', {'lr': 0.1}))
        self.scheduler = getattr(torch.optim.lr_scheduler,
                                 self.get('scheduler/name', 'MultiStepLR'))\
            (self.optimizer, **self.get('scheduler/kwargs',
                                        {'gamma': 0.1, 'milestones': [100, 150]}))
        self.criterion = torch.nn.CrossEntropyLoss()

    def _build_loaders(self):
        # Feel free to hack your own stuff down below.
        # Build the dataloaders
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=self.get('data/root'), train=True,
                                                download=True, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(trainset,
                                                        batch_size=self.get('training/batch_size'),
                                                        shuffle=True,
                                                        num_workers=self.get('data/num_workers', 2))

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(testset,
                                                       batch_size=self.get('training/batch_size'),
                                                       shuffle=False,
                                                       num_workers=self.get('data/num_workers', 2))

    @property
    def device(self):
        return self.get('device', 'cuda:0')

    def train_epoch(self):
        # Feel free to hack in your code below.
        self.model.train()
        # The progressbar (`self.progress`) is provided courtesy of IOMixin, and is based on tqdm.
        for input, target in self.progress(self.train_loader, desc='Training'):
            # Load tensors to device
            input, target = input.to(self.device), target.to(self.device)
            # Evaluate loss, backprop and step.
            prediction = self.model(input)
            loss = self.criterion(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Evaluate metric
            with torch.no_grad():
                error = torch.argmax(prediction, dim=1).eq(target).float().mean()
            # Log if required to. `log_scalars_now` and `log_scalar` is brought to you
            # by `TensorboardMixin`.
            if self.log_scalars_now:
                self.log_scalar('training/loss', loss.item())
                self.log_scalar('training/accuracy', error.item())
            # Tell speedrun that current step is over. This increments the step counter, which you
            # can access with `self.step`. This is not required but strongly recommended,
            # especially if you're using the `TensorboardMixin`.
            self.next_step()

    def validate_epoch(self):
        self.model.eval()
        correct = 0
        total = 0
        for input, target in self.progress(self.test_loader, desc='Validation'):
            input, target = input.to(self.device), target.to(self.device)
            prediction = self.model(input)
            correct += torch.argmax(prediction, 1).eq(target).sum().item()
            total += input.size(0)
        error = correct / total
        # Again, the `log_scalar` method is courtesy of the `TensorboardMixin`.
        self.log_scalar('validation/accuracy', error)

    def checkpoint(self, force=True):
        # You have access to the current epoch count via `self.epoch`. But for this to work, you'll
        # need to call `self.next_epoch` in `self.train` below.
        save = force or (self.epoch % self.get('training/checkpoint_every', 5) == 0)
        if save:
            info = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'config/model/name': self.get('model/name', 'ResNet18'),
                'config/model/kwargs': self.get('model/kwargs', {})
            }
            # `checkpoint_path` is provided by speedrun, and contains the step count. If you do use
            # it, be sure to increment your setp counter with `self.next_step`.
            torch.save(info, self.checkpoint_path)

    def train(self):
        # The progress bar is provided courtesy of `IOMixin`.
        for epoch_num in self.progress(range(self.get('training/num_epochs', 200)), desc='Epochs'):
            self.train_epoch()
            self.validate_epoch()
            self.scheduler.step(epoch_num)
            self.checkpoint(False)
            # The following increments the epoch counter, which is provided by speedrun.
            # The current epoch count is given by `self.epoch`.
            self.next_epoch()
            # The function below is provided by `TensorboardMixin`. It will backup your
            # tensorboard log files as json in the log directory of your experiment.
            self.dump_logs_to_json()


if __name__ == '__main__':
    # Be sure to call `run` and not `train`. Speedrun knows to map a `run` call to `train` via the
    # `DEFAULT_DISPATCH` attribute.
    CifarTrainer().run()