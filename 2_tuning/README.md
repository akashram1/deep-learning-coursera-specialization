# Course 2: 

# Week 3: Tuning & Tensorflow
### Hyperparm Importance
In decreasing order of importance
1. Learning rate α 
2. Momentum β (~0.9), # of hidden units, mini batch size
3. # of Layers, learning rate decay
4. With Adam, tuning params not worth it. β1 ~ 0.9, β2 ~ 0.999 and ε 10e-8 

:star: # units > layers :star:

### How to Search ?
1. `Grid Search`: Good for few hyperparams. Waste on larger scale as some hyperparam changes are not worth it (like ε).
2. `Random Search / Coarse to Fine`: Works efficiently to find regions that work well. Zoom into those and continue search.  

:thought_balloon: Grid search was wisdom 5 years ago at CMU! :thought_balloon:

### Appropriate scale for Hyperparams
:star: # units > Sampling at random `!=` Sampling **uniformly** at random. `=>` Pick scale correctly :star:
1. In some cases sampling randomly will do the job (`for layer in [2,3,4]`)
2. In some case range is so big that sampling randomly will favor some subregions over others.

:star: Search `α in range [10e-4, 1]`. Transform to `log` scale and then search randomly. :star:

3. Exponentially weighted averages:
   1. β = 0.9 is averaging over last 10 gradients. β = 0.999 is last 1K gradients. 
   2. So flip it to `1 - β in [0.1, 0.001]` and do the `log` transformation 

:star:Searching in `log` and not `linear` scale will improve odds you land on range where β sensitivity is high (`0.999 -> 0.9995`)  :star: