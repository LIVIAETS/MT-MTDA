This is the official implementation for WACV 2021 paper of [Unsupervised Multi-Target Domain Adaptation ThroughKnowledge Distillation](https://arxiv.org/abs/2007.07077),

### Requirements

- pytorch <= 1.4.0
- sacred (https://github.com/IDSIA/sacred)

### Datasets

- You can download Office31 and OfficeHome from:
    
        Office-31: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
        Office Home: http://hemanthdv.org/OfficeHome-Dataset/
    
- In order to easily run our code, we expect the datasets for Office31 and OfficeHome to found be at: '~/datasets/', here are how we get the path to our datasets in our code:
        
        a = os.path.expanduser('~/datasets/amazon/images')
        w = os.path.expanduser('~/datasets/webcam/images')
        d = os.path.expanduser('~/datasets/dslr/images')

        Ar = os.path.expanduser('~/datasets/OfficeHome/Art')
        Cl = os.path.expanduser('~/datasets/OfficeHome/Clipart')
        Pr = os.path.expanduser('~/datasets/OfficeHome/Product')
        Rw = os.path.expanduser('~/datasets/OfficeHome/RealWorld')

### Visualization

- Originally, this project used a mongoObserver from sacred in combination with Omniboard (https://github.com/vivekratnavel/omniboard) to visualize experiments. You can either use this or integrate your own visualizer (e.g. visdom)

### Run the code

- All the code is run using sacred, you can see examples in the folder `experiments/`
- Example, to run all of our results on the OfficeHome with AlexNet backbone:
    
        python experiments/exp_kd_multi_target_da_grl_cst_fac.py
- For other experiments you can edit `exp_kd_multi_target_da_grl_cst_fac.py` to change our hyper-parameters

### Evaluation

- Coming soon.

### Credits

We used some of the code from:
  - https://github.com/thuml/CDAN: to load ImageClef dataset
  - https://github.com/jindongwang/transferlearning/tree/master/code/deep/TCP: to load Office31 dataset
