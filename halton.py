from six import print_
def vdc(b,n):
    ri=0.0
    i,l=0,n
    while l>0:
        ri += l % b
        ri *= b
        l  = l // b
        i  += 1
    i += 1
    return ri / b**i
primes = [2,3,5,7,11,13,17,19,23,29]
def eratosthenes(lp):
    global primes
    seeds = range(primes[-1]+2,lp,2)
    for p in primes[1:]:
        seeds = list(filter(lambda n:n%p != 0, seeds))
    while len(seeds) >0:
        primes.append(seeds[0])
        seeds = list(filter(lambda n:n%primes[-1] != 0, seeds))
def initPrimes(n):
    global primes
    lp = len(primes)
    while len(primes) < n:
        #print_("Extending primes ...")
        lp <<=1
        eratosthenes(lp<<1)
    primes = primes[:n]
halton = lambda k:[vdc(p,k) for p in primes]
#initPrimes(1<<8)
#print_(halton(1)[:5]) 
#print_(len(halton(1)) )
