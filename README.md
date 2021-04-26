# fastOrthTrafo
## python code for fast orthogonal transforms for pricing derivatives with quasi-Monte Carlo


According to the paper by Gunther Leobacher,
*Fast orthogonal transforms and generation of Brownian paths*,
Journal of Complexity, Volume 28, Issue 2, April 2012, Pages 278-302,
[https://doi.org/10.1016/j.jco.2011.11.003 ] 
alternatives to principal component analysis and 
Brownian bridge generation or stratified Monte Carlo and quasi-Monte Carlo can
by means of orthogonal transformations provide a faster, i.e., O(n log(n)) 
convergence.

The files in the repository provide these transformations in python code
and showcase their use in example payoffs for several path types.

* orthoTrafos.py       : collects the transformations
* payoffs.py           : assembles several payoff functionals and path types
* halton.py            : provides simple quasi-random points (Halton sequence) for payoffExample*
* payoffExample.py     : demonstrates the use and plots a graph using matplotlib
* payoffExamplesTK.py  : uses a simple (Tk) graphic interface for the demonstration

Versions used:
* Python  : 3.9.4
* SciPy   : 1.6.2
* NumPy   : 1.20.1
For the example files:
* matplotlib : 3.4.1
