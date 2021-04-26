# versions used:
# (Python  : 2.7.16 # six : 1.15.0 )
# Python3 : 3.9.4
# numpy : 1.20.1
# scipy : 1.6.2

import numpy           as np
import scipy           as sp
import scipy.special   as spsp
from   scipy.stats import norm as spnorm
import scipy.integrate as spint
#from random import random 

### overall stepsize etc parameters

# s needs to be in [1..8]
s = 6;
# n = no of steps (dimension)
n = 1<<s;

r = 0.045; sigma = 0.3; T = 1.0;
s0 = 100; k = 100;

dt = T/n; strike = 95;
time = np.array(range(1,n+1))*dt;

#(* For Barrier options *)
upper = 120; lower = 80;

phi = 0.00; # for Papageorgious option
phi = 0.3;
phi = 1.05 *3*np.pi/4;
phi = 2*np.pi/3;  # for the variant of Papageorgious option


### distribution auxiliaries

invNormalCDF = lambda xlist: [spsp.ndtri(el) for el in xlist];
npdf = lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi);
fnig = lambda x,a,b,d,m: a/np.pi *np.exp(d*np.sqrt(a*a-b*b)+b*(x-m))*\
    d*spsp.kv(1,a*np.sqrt(d*d+(x-m)**2))/np.sqrt(d*d+(x-m)**2) ;
fnig1  = lambda x : fnig(x, 18.3, -1.06, 0.0184, 0.000434);
Gi0    = spsp.ndtri( spint.quad(fnig1,-sp.inf,0)[0] );
#Gnig  #################### TODO by Runge-Kutta :
#Gnig=spint.odeint(lambda y,t:spnorm.pdf(t)/fnig1(y),0,np.linspace(0,1,100))
# Using polynomial fit instead
Gnig = lambda x : 0.0006160965767867249 + 0.01771276305996152*x - 0.0008919398978123955*x**2 + 0.004394149716934829*x**3 - 0.000011597675637392617*x**4 - 0.00003075893982138245*x**5
emunig  = spint.quad( lambda x:np.exp(x)    *fnig1(x), -sp.inf, sp.inf)[0];
munig   = spint.quad( lambda x:       x     *fnig1(x), -sp.inf, sp.inf)[0];
signig2 = spint.quad( lambda x:(x-munig)**2 *fnig1(x), -sp.inf, sp.inf)[0];

chyp = lambda a, b, d, m : \
  np.sqrt(a*a - b*b)/(2*a*d*spsp.kv(1, d*np.sqrt(a*a - b*b)));
fhyp = lambda  x, a, b, d, m  : \
  chyp(a, b, d, m)*np.exp(-a*np.sqrt(d*d + (x - m)**2) + b*(x - m));
fhyp1 = lambda  x  : fhyp(x, 30, 0, 0.01, 0.001);
Gih0   = spsp.ndtri( spint.quad(fhyp1,-sp.inf,0)[0] );
#Ghyp  #################### TODO by Runge-Kutta :
#Ghyp=spint.odeint(lambda y,t:spnorm.pdf(t)/fhyp1(y),0,np.linspace(0,1,100))
# Using polynomial fit instead:
Ghyp = lambda x :  0.0010001512700608768 + 0.05014651120552324*x  + 0.0018029496372113066*x**3
emuhyp  = spint.quad( lambda x:np.exp(x)    *fhyp1(x), -sp.inf, sp.inf)[0];
muhyp   = spint.quad( lambda x:       x     *fhyp1(x), -sp.inf, sp.inf)[0];
sighyp2 = spint.quad( lambda x:(x-muhyp)**2 *fhyp1(x), -sp.inf, sp.inf)[0];

a = [[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]];
amul = lambda xy: [a[0][0]*xy[0]+a[0][1]*xy[1],a[1][0]*xy[0]+a[1][1]*xy[1]];
def lenlog2(xlist):
    n = len(xlist)
    i = -1
    while n>0:
        n>>=1
        i+=1
    return i
def OTP( data ):
    trans = data
    n = len(data)
    n2 = n >> 1
    while n>1:
        trans = np.array([amul(el) for el in \
          np.array(trans).reshape(n2,2)]).transpose().flatten()
        n>>=1
    return trans

sdevs = np.array([np.std(np.cumsum(np.append([0],OTP(el))),ddof=1) for el in np.identity(n)]);
perm1 = list(reversed([int(el[1]) for el in sorted([list(el) for el in np.array([sdevs,range(n)]).transpose()])]));
otpperm = list(([int(el[1]) for el in sorted([list(el) for el in np.array([perm1,range(n)]).transpose()])]));
pOTP = lambda xlist : np.insert(np.cumsum(OTP(xlist)),0,0)/np.sqrt(n);


##(* Path generation methods *)
BlackScholesPath = lambda nvec : \
 [s0*np.exp(el) for el in sigma*np.cumsum(nvec)/np.sqrt(n) + (r - sigma**2/2)*time]
NIGPath = lambda nvec : \
 [s0*np.exp(el) for el in np.cumsum([Gnig(el2) for el2 in nvec]) + (r - np.log(emunig)*dt)*time]
HypPath = lambda nvec : \
 [s0*np.exp(el) for el in (np.cumsum([Ghyp(el2) for el2 in nvec]) + (r - np.log(emuhyp)*dt)*time)]


#### some simple payoffs 

#(*European call*)
payoffCall = lambda path : np.exp(-r*T)*max(path[-1] - k, 0);
#(*Delay (call) option*)
payoffDelay = lambda path : np.exp(-r*T)*max(path[-1] - path[n//2], 0);
#(*Lookback call*)
payoffLookbackCall = lambda path : np.exp(-r*T)*max(path[-1] - min(path), 0);
#(*Lookback put*)
payoffLookbackPut = lambda path : np.exp(-r*T)*max(-path[-1] + max(path), 0);
#(*Fixed strike Asian *)
payoffAsian = lambda path : np.exp(-r*T)*max(T/n *sum(path) - strike, 0);
#(*Floating strike Asian *)
payoffFloatingStrike = lambda path : \
  np.exp(-r*T)*max(path[-1] - T/n *sum(path), 0);
#(*Double Asian *)
payoffDoubleAsian = lambda path : \
  np.exp(-r*T)*max(max([(el[0] - el[-1])*2*T/n for el in [ \
     [sum(el2) for el2 in np.array(path).reshape(2,len(path)>>1) \
     ]] ]), 0);

#(* Helper functions for option payoffs *)
IsPositive = lambda x : 1 if x  > 0 else 0 ;
pPart = lambda x : (abs(x) + x)/2;

#(*Weighted Asian option*)
#(*weight=Table[Cos[2 Pi k/n]dt,{k,1,n}];*)

its = lambda x :\
  np.append(-np.diff(x),x[-1]) * np.sqrt(len(x));
weight = [dt]*n;

weight = its(OTP([1]+[0]*(n-1)));
weight = weight/sum(weight);
payoffWeightedAsian = lambda path:\
  np.exp(-r*T)*max(sum(np.array(weight)* np.array(path)) - strike, 0);
payoffWeightedGeometricAsian = lambda path:\
  np.exp(-r*T)*max(np.exp(sum(np.array(weight)* np.array(path))) - strike, 0);

#list_plot(weight)
#plot(lambda x: payoffWeightedAsian( BlackScholesPath( OTP([0]*(n-1)+[x]))),  -10, 10)

#(*Digital Option like in Papageorgiou*)
payoffPapa = lambda path: np.exp(-r*T)* \
    sum( (lambda xl: [IsPositive(el) for el in np.diff(xl)]*xl[1:])(np.append([s0],path)) )/n;
payoffPapa = lambda path:\
  sum( (lambda xl: [IsPositive(el) for el in np.diff(xl)]*xl[1:])(np.append([s0],path)) )/n;

#(*Variant of Papageorgiou's option*)
lag = 2;
payoffPapaVar = lambda path: np.exp(-r*T)* \
    sum( (lambda xl: [IsPositive(el) for el in (xl[l:]-xl[:-l])]*xl[lag:])(np.append([s0],path)) )/n;

#(*Up and out barrier option*)
payoffUpAndOut = lambda path:\
 np.exp(-r*T)* ( pPart(path[-1] - strike) if max(path) <= upper else 0);
#(*Up and in barrier option*)
payoffUpAndIn = lambda path:\
 np.exp(-r*T)* ( pPart(path[-1] - strike) if max(path) >= upper else 0);
#(*Down and out barrier option*)
payoffDownAndOut = lambda path:\
 np.exp(-r*T)* ( pPart(path[-1] - strike) if max(path) >= lower else 0);
#(*Down and in barrier option*)
payoffDownAndOut = lambda path:\
 np.exp(-r*T)* ( pPart(path[-1] - strike) if max(path) <= lower else 0);
#(*Double knock out barrier option*)
payoffDoubleKnockOut = lambda path:\
 np.exp(-r*T)* ( pPart(path[-1] - strike) if min(path) >= lower and max(path) <= upper else 0);

allpayoffs=[\
payoffCall, payoffDelay, payoffLookbackCall,\
payoffLookbackPut, payoffAsian,\
payoffFloatingStrike, payoffDoubleAsian,\
payoffWeightedAsian, payoffWeightedGeometricAsian,\
payoffPapa, payoffPapaVar,\
payoffUpAndOut, payoffUpAndIn,\
payoffDownAndOut, payoffDownAndOut,\
payoffDoubleKnockOut]

allpayofflegs=[\
"Call", "Delay", "LookbackCall",\
"LookbackPut", "Asian",\
"FloatingStrike", "DoubleAsian",\
"WeightedAsian", "WeightedGeometricAsian",\
"Papa", "PapaVar",\
"UpAndOut", "UpAndIn",\
"DownAndOut", "DownAndOut",\
"DoubleKnockOut"]
