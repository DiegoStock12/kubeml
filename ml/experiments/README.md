# KubeML Experiments

In these experiments we compare the performance of running training workflows in KubeML and compare it to a Tensorflow
distributed session. The main points of focus of the experiments are:

1. Application Related Factors:
    * Time to accuracy
    * Time per epoch as a function of batch
    * Max accuracy achievable
    * Local SGD test bench: see how parallelism and number of local update affect performance (time and accuracy)
    * We will employ well-known networks and datasets for this matter: LeNet, VGGNet, Resnet-18, CIFAR, MNIST...
    
2. Provider Related Factors:
    * Usage of resources: See if being able to pack many network in the same GPU improves performance and usage
    * Resource density: How many network we can train in parallel efficiently with limited hardware
    * Scalability: Degradation of performance with parallelism
    * Resiliency: use of chaos monkey to inject failures in functions
    
3. Developer Related Factors:
    * Number of lines of code needed to go from a local deployment to a distributed training environment
    * Cost of training when compared to a tensorflow server