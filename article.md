# How to upgrade your PyTorch model to run on AWS Sagemaker

## Intro

AWS Sagemaker makes it easy to train and deploy a machine learning model in the cloud. Sagemaker is an AWS service built on top of the Elastic Container Service (ECS) and S3. You write a training script and provide your dataset, and Sagemaker uploads them to S3, trains your model, and saves your trained model to S3 to be downloaded or deployed directly using Sagemaker Endpoints.

Training your model on Sagemaker gives you access to a range of powerful machines, and the ability to distribute your training across as many as you want! If you use Sagemaker’s power to the max though, you may end up paying a lot for it, so watch out.

In this article, we’ll quickly go through the steps to upgrade your normal PyTorch training script, into a Sagemaker-compatible script, that can be distributed over multiple GPUs.

## Upgrade your training script

To start with we’ve got our normal PyTorch training file, with a model and a training function that saves the model at the end. You can see the one I'm using [here](https://github.com/aydin-utting/upgrade-pytorch-to-sagemaker-article/blob/main/normal_script.py). We could run this in a notebook, or an iPython terminal. We only need to make a few changes to make this script run on AWS.

Firstly, our train function needs to be run within the top-level code environment. We do this by putting the code within a `if __name__ == ‘__main__’` block. When we run the python file from the terminal, the variable `__name__` gets set to `”__main__”`. This means that we can run

```
python -m training_script.py
```

And it will run the block contained within `if __name__ == '__main__'`, but if we import some functions from our file like this

```python
from training_script import my_funcmy_func()
```

Then the block will not be run, because here `training_script.__name__ = “training_script”`.

We care about this because if we use our trained model in a SageMaker Endpoint, Sagemaker will import our model from our training script file, and we don’t want our training script to run every time! Take a look at the Python docs [here](https://docs.python.org/3/library/__main__.html) to learn about the Python top-level code environment.

```python
If **name** == “**main**”:
train()
```

When Sagemaker runs our endpoint, it is passed the training hyperparameters as arguments. The standard way to parse these is to use the `argparse` library. We also have access to loads of environment variables containing information about the EC2 instance we are running on, such as how many GPU cores are available, or where to save the trained model.

Our training script will be run on a Docker container on AWS. Sagemaker will expect our data to be in a specific place when it runs our training script, and it provides that path in the SM_CHANNEL_TRAINING environment variable. We can actually specify any SM_CHANNEL_XXXX and put different data in each, such as SM_CHANNEL_VALIDATION or SM_CHANNEL_TEST.

When our training job is finished, Sagemaker will upload our trained model to S3. To do this, it expects it to be the directory specified by the MODEL_DIR environment variable.

```python
if **name** == "**main**":
    parser = argparse.ArgumentParser() # hyperparameters:
    parser.add*argument("--batch-size",type=int,default=64)
    parser.add_argument(“--test-batch-size", type=int, default=1000)
    parser.add_argument(“--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    # directories to save the model and get the training data:
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ[“SM_CHANNEL_TRAINING"])
    *, args = parser.parse_args()
    train(args)
```

We can then update our train function to use these hyperparameters, such as `args.batch_size`. We also need to update our dataset loaders to get the data from the directory `args.data_dir`, and our saving function to save to args.model_dir.

We could stop here if we wanted, and this script is ready to be used on AWS! That being said, we’re not taking advantage of a big feature of Sagemaker: distributed training.
Distributed Training
One of the powers of the cloud is horizontal scaling: the ability to increase the number of machines that are running your code. Sagemaker’s built-in models are built to take advantage of this, and you can specify the number of instances you want to run your training script on. We can build this into our PyTorch model, and speed up training!

We need to use the `pytorch.distributed` package to parallelise our training loop. It’s surprisingly simple to get your model running on multiple instances at the same time using PyTorch’s DistributedDataParallel.

Some new imports:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
```

And we need to add some new arguments in our top level environment:

```python
parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
parser.add_argument("--backend",type=str,default=None)
```

- `hosts` is an array of the names of all the instances that our training script is running on
- `current_host` is the name of the host that is currently running the code
- `num_gpus` is the number of gpu’s on the instance
  Backend is the PyTorch distributed backend that we want to use (see [here](https://pytorch.org/docs/stable/distributed.html) )

Inside the `train` function, we set the device to “cuda” if we have some gpu’s available:

```python
use_cuda = args.num_gpus > 0
device = torch.device(“cuda” if use_cuda else “cpu”)
```

We’ll also want to put the data onto the GPU if there’s one available using `data, target = data.to(device) ,target.to(device)` in the test and train loops.

Then we need to start a process group on each of our machines, we tell PyTorch how many instances there are (`world_size`) and where the current process sits in that list (`host_rank`) so that it can link them up during training and average gradients between machines.

```python
world_size = len(args.hosts)
os.environ["WORLD_SIZE"] = str(world_size)
host_rank = args.hosts.index(args.current_host)
os.environ["RANK"] = str(host_rank)
dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
```

We then need to wrap the model in DistributedDataParallel and set it to the device

```python
model = Net().to(device)
model = DDP(model)
```

We need to split the training data across our different instances. To do this we use a DistributedSampler. The sampler splits the dataset for us across the different instances.

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set, num_replicas=dist.get_world_size(), rank=dist.get_rank()
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=False
)
```

Now we’ve got a distributed model!

### Logging

When you’re training on your own computer or in an environment you can access directly, you can use python’s built in `print` function to keep tabs on your model in the terminal. On Sagemaker, `print` statements will print to the docker container’s `stdout` and you’ll never see it! To get logging working on Sagemaker we just need to add:

```python
import logging
import sys
logger = logging.getLogger(**name**)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
```

We can now use `logging.info(...)` everywhere we would normally use `print(…)`, and we’ll see it in the Jupyter notebook!

### Model Loading

Now our simple training script is ready to run on Sagemaker! We could start straight away, and our trained model would be saved to `model_dir/model.pth`. However we'd have to go to the S3 bucket we saved it in and retrieve it manually. In order to deploy our model with Sagemaker ndpoints, we need to provide a function to load that model from storage.

```python
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
    model.load_state_dict(torch.load(f))
    return model.to(device)
```

This function receives the model_dir, and we create a `Net()` object and load in our saved hyperparameters.

Now we are ready to go! You can see the completed code, with all these changes [here](https://github.com/aydin-utting/upgrade-pytorch-to-sagemaker-article/blob/main/normal_script.py).

## Conclusion

You’ve just upgraded your training script to work on AWS Sagemaker, and take advantage of its distributed machine learning power!
Training your PyTorch model on AWS is great if you need a lot of power. You’ve now got an AWS compatible training script, that can run distributed across many instances. There’s a little more work to be done to start your training job, which we can do using a Sagemaker Notebook instance.
