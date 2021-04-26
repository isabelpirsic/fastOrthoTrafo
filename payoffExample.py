from six import print_
import matplotlib.pyplot as plt
from random import random 
from time import time as AbsoluteTime

from orthoTrafos import HaarInverse, DPCA, Walsh, DCT1
from halton import *
from payoffs import *

### setting up a QMC method

unifctr = -1
def seedunif(k):
    global unifctr
    unifctr = k
initPrimes(n)
def haltonIter():
    global unifctr
    unifctr += 1
    return halton(unifctr)

#(* Parameters for the MC-Simulation *) #######
# see also near end of file
mmin = 2; mmax = 6;

###(* Parameters for the MC-Simulation *) #######
methleg = {"Forward", "PCA", "BB","Walsh","DCT1"};
Forwd = lambda x:x;
methods = lambda xl : [Forwd(xl), DPCA(xl), HaarInverse(xl), Walsh(xl), DCT1(xl)];
nmethods = len(methods([0.]*n));
payoff = payoffAsian;
theNormal = invNormalCDF;
thePath = HypPath;
theRNGMethod = haltonIter

##### main loop of calculating payoffs according to the different path constructions
##### (i.e., according to the different orthohogonal transformations)
sdev = []; mvec = [];
#SeedRandom[2345];
for m in range(mmin,mmax+1):
    oldtime = AbsoluteTime()
    runs = 1<<s; runs1 = 1<<m;
    mean = np.array([0.]*nmethods)
    mean2= np.array([0.]*nmethods)
    for j in range(runs):
        seedunif(0)
        mean1 = np.array([0.]*nmethods) ;
        randshift = np.array([random() for _ in range(n)]);
        for l in range(runs1):
            nsample =  theNormal([ el% 1 for el in (theRNGMethod() + randshift)]);
            sample = methods(nsample);
            paths = [thePath(el) for el in sample];
            mean1 += np.array([payoff(el) for el in paths]) ;
        mean1  = mean1/runs1;
        mean  += mean1;
        mean2 += mean1**2;
    mean  = mean/runs;
    mean2 = mean2/runs;
    sdev = sdev + [np.sqrt(mean2 - mean**2)]
    mvec = mvec + [m] ;
    newtime = AbsoluteTime();
    print_("m = ", m, "\nTime : ", newtime - oldtime);
    print_("Averages : ", mean, "\nStdDevs : ", np.sqrt(mean2 - mean**2));
sdev=np.array(sdev).transpose()

### plot of the results
#methods = [Forwd(xl), DPCA(xl), HaarInverse(xl), Walsh(xl), DCT1(xl)];
labels = ["Forward", "DPCA", "BB", "Walsh", "DCT1"];
colors=["blue","red","green","yellow","black","violet"]
#sum(list_plot( np.array([mvec,np.log2(sdev[i])]).transpose()  ,plotjoined=true, color=colors[i])for i in range(nmethods) )
plt.clf()
for i in range(nmethods):
    plt.plot( range(mmin,1+mmax),np.log2(sdev[i]), color=colors[i], label=labels[i]  )
plt.legend()
plt.show()
