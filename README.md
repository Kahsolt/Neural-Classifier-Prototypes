# Neural-Classifier-Prototypes

    Evolving from a pure-color image or random noise towards class-wise inherent prototypes for a neural classifier

----

For a well-trained neural classifier `f`, we call an input `x` to be a prototype of class `y` iff. `f(x) == y st. d(f(x)) / d(x) == 0`. 
Which means this input `x` is a extremum of `f`, causing zero gradient on `f`, and reasonably correctly classified by `f` (with the highest confidence). 

Due to the inherent nature of NN models, a 1000-class classifier may have much more than 1000 class-wise prototypes, constrasting with centroids-based clustering models. 
Even though, there are much much moooore inputs that are NOT exactly equal to any prototype, and they will cause a non-zero gradient on `f`.

An adversarial attacker usually seeks along the direction of max gradient `max. d(f(x)) / d(x)` for a small perturbation `dx` letting `f(x + dx) != y`, using the naive FGSM or PDG method. 
We now consider about the opposite: for any given input `x`, find its closest class-wise prototypes inherent of `f`. 
In addition, we are quite curious about this: when `x` starts evolving/alientnating from a total blank or guassian noise, **to what degree** will `f` start to recognize it (with a high confidence), is that generated texture also recognizable for a human-being?

Yeah, you might've got it - we are treating adversarial attacks as kind of generative model, and the frozen classifier is its discriminator counterpart ;)


### Experiments

#### Alienate from blank / pure color image

Occupying any label-ignorant gradient-based adversarial attack (we try `PGD`, `MIFGSM` and `PGDL2`), for each input `x` towards then given classifier `f`: 

  - Find perturbations `dx_{i}` st. `f(c + dx_{i}) == y_{i} st. d(f(x + dx_{i})) / d(x + dx_{i}) -> 0`, where `c` is a const
  - Find perturbations `dx_{i}` st. `f(c + dx_{i}) == y_{i} st. | d(f(x + dx_{i})) / d(x + dx_{i}) | -> inf`

We test model `f=resnet18` (pretrained over the whole ImageNet), victim dataset `X=imagenet-1k`, attack setting `atk=('pgd', 0.1, 0.001)`


#### Alienate from random noise

For each input `x` towards then given classifier `f`: 

  - Find perturbations `dx_{i}` st. `f(r + dx_{i}) == y_{i} st. d(f(x + dx_{i})) / d(x + dx_{i}) -> 0`, where `r` follow some kind of stochastic distribution
  - Find perturbations `dx_{i}` st. `f(r + dx_{i}) == y_{i} st. | d(f(x + dx_{i})) / d(x + dx_{i}) | -> inf`

We test model `f=resnet18` (pretrained over the whole ImageNet), victim dataset `X=imagenet-1k`, attack setting `atk=('pgd', 0.1, 0.001)`


#### Alienate from a given picture

  - Find perturbations `dx_{i}` st. `f(x + dx_{i}) == y_{i} st. d(f(x + dx_{i})) / d(x + dx_{i}) -> 0`, where `i` enumerates over all target classes of `f`
  - Find perturbations `dx_{i}` st. `f(x + dx_{i}) == y_{i} st. | d(f(x + dx_{i})) / d(x + dx_{i}) | -> inf`

We test model `f=resnet18` (pretrained over the whole ImageNet), victim dataset `X=imagenet-1k`, attack setting `atk=('pgd', 0.1, 0.001)`


```
log\<model>_<train_dataset>-<mode>-<atk_dataset>_<method>_e<eps>_a<alpha>.pkl
resnet18_imagenet-min-svhn_pgd_e3e-2_a1e-3
```

----

by Armit
2022/10/09 
