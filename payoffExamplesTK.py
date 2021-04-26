from tkinter import *
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib import ticker
from time import time as AbsoluteTime

#### 
from random import random 
from orthoTrafos import *
from halton import *
from payoffs import *

#(* Parameters for the MC-Simulation *) #######
# see also near end of file

methodfs=[Fwd]
methleg=["Forward"]
payoff = allpayoffs[0]
mmin = 2
mmax = 6

def retrieve():
    global methodfs, methleg, mmin, mmax, payoff
    methodfs = []
    methleg = []
    for i in range(len(allmethods)):
     if methVar[i].get()==1:
      methodfs += [allmethods[i]]
      methleg  += [allmethleg[i]]
    for i in range(len(allpayoffs)):
     if Combo.get()==allpayofflegs[i]:
      payoff = allpayoffs[i]
    mmin = int(mminBtn.get())
    mmax = int(mmaxBtn.get())
    root.quit()
    root.destroy()
 
root = Tk()
root.title("Set up example")
root.geometry("500x300")
frame  = Frame(root, bg = "gray" )

Combo = ttk.Combobox(frame, values = allpayofflegs)
Combo.set("Pick a payoff")
Combo.pack(padx = 5, pady = 5)

frame1 = Frame(frame, bg="black")
frame1.pack(padx = 5, pady = 15,side=LEFT)
frame2 = Frame(frame, bg="black")
frame2.pack(padx = 5, pady = 15,side=LEFT)
frame3 = Frame(frame, bg="black")
frame3.pack(padx = 5, pady = 15,side=LEFT)
frame5 = Frame(root, bg="blue")
frame5.pack(padx = 5, pady = 15,side=RIGHT)
frame4 = Frame(root, bg="black")
frame4.pack(padx = 5, pady = 15,side=RIGHT)
frame.pack(padx = 5, pady = 15,side=LEFT)

methVar = [0 for _ in allmethods]
ChkBttn = [ Checkbutton() for _ in allmethods] 

for i in range(6):
 methVar[i] = IntVar()
 ChkBttn[i] =  Checkbutton(frame1, text = allmethleg[i], variable = methVar[i], \
 relief = SOLID )
 ChkBttn[i].pack(padx = 5, pady = 5)
methVar[0].set(1) 

for i in range(6,10):
 methVar[i] = IntVar()
 ChkBttn[i] =  Checkbutton(frame2, text = allmethleg[i], variable = methVar[i], \
 relief = SOLID )
 ChkBttn[i].pack(padx = 5, pady = 5)
methVar[0].set(1) 

for i in range(10,14):
 methVar[i] = IntVar()
 ChkBttn[i] =  Checkbutton(frame3, text = allmethleg[i], variable = methVar[i], \
 relief = SOLID )
 ChkBttn[i].pack(padx = 5, pady = 5)
methVar[0].set(1) 

Button = Button(frame5, text = "Submit", command = retrieve, fg="green")
Button.pack(padx = 5, pady = 5)

mminLbl = Label(frame4, text=("Minimum m :"))
mminLbl.pack()
mminBtn = Entry(frame4, width = 15)
mminBtn.insert(0,str(mmin))
mminBtn.pack(padx = 5, pady = 5)
 
mminLbl = Label(frame4, text=("Maximum m :"))
mminLbl.pack()
mmaxBtn = Entry(frame4, width = 15)
mmaxBtn.insert(0,str(mmax))
mmaxBtn.pack(padx = 5, pady = 5)
 
root.mainloop()
#print(0/0)
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


###(* Parameters for the MC-Simulation *) #######
#methleg = ["Forward", "PCA", "BB","Walsh","DCT1"];
#Fwd = lambda x:x;
#methods = lambda xl : [Fwd(xl), DPCA(xl), HaarInverse(xl), Walsh(xl), DCT1(xl)];
methods = lambda xl : [meth(xl) for meth in methodfs]
nmethods = len(methods([0.]*n));
#payoff = payoffAsian;
theNormal = invNormalCDF;
thePath = HypPath;
theRNGMethod = haltonIter

##### main loop of calculating payoffs according to the different path constructions
##### (i.e., according to the different orthohogonal transformations)
sdev = []; mvec = [];
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
    print("m = ", m, "\nTime : ", newtime - oldtime);
    print("Averages : ", mean, "\nStdDevs : ", np.sqrt(mean2 - mean**2));
sdev=np.array(sdev).transpose()

### plot of the results
#methods = [Fwd(xl), DPCA(xl), HaarInverse(xl), Walsh(xl), DCT1(xl)];
labels = methleg #["Forward", "DPCA", "BB", "Walsh", "DCT1"];
colors= ['C'+str(i) for i in range(nmethods)] #["blue","red","green","yellow","black","violet"]
plt.clf()
fig,ax=plt.subplots()
for i in range(nmethods):
    plt.plot( range(mmin,mmax+1),np.log2(sdev[i]), color=colors[i], label=labels[i]  )
#fig.set_title("Payoff : "+Combo.get())
plt.legend()
plt.show()
