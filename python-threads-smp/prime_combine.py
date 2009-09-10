#!/usr/bin/env python

import sys
import os
import math
import thread
import multiprocessing
import string
import time

from timeit import Timer

import dyn_options


""""

Original version, lifted from ....
Notice that all threads share nexti and prime..

"""

def help() :
    print """
    This program is used to time various threading/multi processing
    implementations of a simple prime search.
    -top      : upper limit of the search (an integer)
    -nthreads : number of threads /processes to use
    -help     : this help screen

    """

def defaults() :
    return dict([("limit", 100), ("nthreads", 1), ("repeat", 5)])

def count_primes(prime) :
    p = reduce(lambda x, y: x + y, prime) - 2 
    return p

def dowork_orig(tn): # thread number tn
    global n,prime,nexti,nextilock,nstarted,nstartedlock,donelock, work
    donelock[tn].acquire()
    nstartedlock.acquire()
    nstarted += 1
    nstartedlock.release()
    lim = math.sqrt(n)
    ops = 0 #added
    nk = 0
    while 1:
        nextilock.acquire()
        k = nexti
        nexti += 1
        nextilock.release()
        if k > lim: break
        #print k
        nk += 1
        if prime[k]:
            r = n / k
            for i in range(2,r+1):
                #print k, i, k*i
                ops += 1 #added
                prime[i*k] = 0
    work[tn] = nk            
    #print 'thread', tn, ' processed', nk, 'values of k; ops ', ops    
    donelock[tn].release()

def main_orig(top, nthreads):
    global n,prime,nexti,nextilock,nstarted,nstartedlock,donelock, work

    n        = int(top)
    nthreads  = int(nthreads)
    prime = (n+1) * [1]
    work  = nthreads * [-99]

    nstarted = 0
    nexti = 2
    
    nextilock = thread.allocate_lock()
    nstartedlock = thread.allocate_lock()
    donelock = []
    for i in range(nthreads):
        d = thread.allocate_lock()
        donelock.append(d)
        thread.start_new_thread(dowork_orig,(i,))
    while nstarted < nthreads: pass
    for i in range(nthreads):
        donelock[i].acquire()

    return count_primes(prime)    

"""
Remove shared variable nexti; a little faster !
Uses naiave load balancing which works because prime 
is shared..
"""

def dowork2(n, nexti, prime) :
    nk    = 0
    step  = -1
    k     = nexti[0]
    lim   = nexti[1]
    if nexti[0] > nexti[1] : 
        raise "boundaries out-of-order"

    while 1 :

        if not (k < lim) : break
        nk += 1

        if prime[k] == 1 :
            r = n / k
            for i in range(2, r+1) :
                prime[i*k] = 0

        k   = k + 1

    #print 'thread exiting; processed ', nk, 'values of k'
    return prime 


def dowork_th(tn, donelock, n, nexti) :
    global prime
    prime = dowork2(n, nexti, prime)
    donelock.release()


def load_balance(s, th) :

    len_s = len(s)
    if len_s == 0 :
        return [ s ]
    
    base = len_s / th
    rem  = len_s - base * th
    K = map(lambda i : i * base, range(1, th+1))
    t = range(1, rem + 1 )  + (th - rem )*[rem]
    K = map(lambda p : p[0] + p[1], zip(K, t))
    K = zip([0] + K, K)
    last = s[len_s - 1]
    s.append(last+1)
    K = map(lambda p : (s[p[0]], s[p[1]]), K) 
    return K

def start_th(fn, args) :
    return  thread.start_new_thread(fn, args)

def main_thread(top, nthreads) :
    global prime
    n             = int(top)
    nthreads      = int(nthreads)
    
    prime         = (n + 1)  * [1]
    donelock      = map(lambda l : l.acquire() and l, 
                        map(lambda i : thread.allocate_lock(), range(nthreads)))

    lim   = int(math.sqrt(n)) + 1
    nexti = range(2, lim, 1) 

    B = load_balance(nexti, nthreads)

    map(lambda i : start_th(dowork_th, (i, donelock[i], n, B[i])),
        range(nthreads))
    
    map(lambda i : donelock[i].acquire(), range(nthreads) )

    return count_primes(prime)    

"""
Using process Pools; No shared variables; 
Load balancing tries to keep the number of operations
per process constant

"""

def dowork_smp(args) :
    n, nexti, chunks = args
    nk    = 0
    ops   = 0
    k     = nexti[0]
    L     = ( n + 1) * [1]
    lim   = len(nexti)
    while 1 :

        k  = nexti[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

    #print 'thread exiting;', 'processed ', nk, 'values of k, ops', ops
    

    len_L = n + 1
    split = len_L / chunks
    
    K     = range(0, len_L - split + 1, split)+[len_L]
    Z     = [ L[k[0]:k[1]] for k in zip(K, K[1:]) ]

    return Z


def smp_load_balance(th , n) :

    def operations(t) :
        return int((n / t) + 1 - t)

    def find_min(thr_alloc) :
        min, lst = thr_alloc[0]
        if min == 0 :
            return 0
        midx = 0
        for index in range(1, len(thr_alloc)) :
            count, lst = thr_alloc[index]
            if count < min :
                min   = count
                midx  = index
        return midx

    lim           = int(math.sqrt(n)) + 1
    nexti         = range(2, lim, 1)

    if th < 2 :
        return [nexti]

    thr_allocs = map(lambda i : (0, [] ), range(th))
    Z = map(operations, nexti)

    L = zip(map(operations, nexti), nexti)

    for i in L :
        ops, index = i
        mindex = find_min(thr_allocs)
        cnt, lst = thr_allocs[mindex]
        cnt += ops
        lst.append(index)
        thr_allocs[mindex] = (cnt, lst)

    #print thr_allocs    
    return map(lambda p: p[1], thr_allocs)

def list_reduce(l1, l2) :
    return map(lambda p : p[0]*p[1], zip(l1,l2))

def reduce_chunk(C) :
    return reduce(lambda x, y : x + y, reduce(list_reduce, C))     

def transpose(K) :
    nthreads = len(K)
    chunks   = len(K[0])
    X = [ (l, k) for k in range(0, chunks) for l in range(0, nthreads)]
    len_X = len(X)
    S  = [ X[k:k+nthreads] for k in range(0, len_X, nthreads)]
    PR = [ [ K[p[0]][p[1]] for p in S[s]] for s in range(0, chunks) ]
    return PR

def main_smp(top, nthreads) :

    n             = int(top)
    nthreads      = int(nthreads)

    prime         = (n + 1) * [1]

    
    B     = smp_load_balance(nthreads, n)
    p     = multiprocessing.Pool(16*nthreads)
    K     = p.map(dowork_smp, map(lambda lst : (n, lst, 16*nthreads), B))
    PR    = transpose(K)
    prime = p.map(reduce_chunk, PR)


    return count_primes(prime)    


"""
Introduce shared variable in the smp module...

"""

def dowork_smp_shared(args) :
    n, nexti, L = args
    nk    = 0
    ops   = 0
    k     = nexti[0]
    lim   = len(nexti)
    while 1 :

        k  = nexti[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

    return []


def main_smp_shared(top, nthreads) :

    n             = int(top)
    nthreads      = int(nthreads)

    manager = multiprocessing.Manager()

    L_m     = manager.list((n+1) * [1])
    B       = smp_load_balance(nthreads, n)
    p       = multiprocessing.Pool(nthreads)

    K       = p.map(dowork_smp_shared, map(lambda lst : (n, lst, L_m), B))
    PR      = transpose(K)
    prime   = p.map(reduce_chunk, PR)


    return count_primes(L_m)    

"""
Uses processes, and shared data...

"""

def create_process(argv) :
    return multiprocessing.Process(target=dowork_smp_shared_2, args=(argv,))

def dowork_smp_shared_2(args) :
    n, nexti, L = args
    nk    = 0
    ops   = 0
    k     = nexti[0]

    lim   = len(nexti)
    while 1 :

        k  = nexti[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

    #print 'thread exiting;', 'processed ', nk, 'values of k, ops', ops
    
    return L


def main_smp_shared_2(top, nthreads) :
    n             = int(top)
    nthreads      = int(nthreads)

    prime         = (n + 1) * [1]

    B     = smp_load_balance(nthreads, n)
    arr   = multiprocessing.Array('i',prime)
    
    procs = map(create_process, map(lambda lst : (n, lst, arr), B))
    map(lambda p : p.start(), procs)
    map(lambda p : p.join(), procs)
    prime = arr[:]
    
    return count_primes(prime)
    
    
"------------------------------------------------"
def time_routine_header() :
    return "description   nthreads   repeat    limit       time (sec)\n"

def time_routine(desc, args) :

    (limit, nthreads, repeat) = args
    s = "%s(%s, %s) " % (desc, limit, nthreads)

    i = "from __main__ import %s" % (desc)

    t = Timer(s, i)
    
    return (desc, nthreads, repeat, limit, t.timeit(int(repeat)))


def repr_tuple(sep) :

    def p (tuple) :
        T = map(lambda l : str(l), tuple)
        return string.join(T, sep)
    return p


def print_single(option) :
    if not option.time_it :
        cnt = main_orig(option.limit, option.nthreads)
        print "number of primes :", cnt
        cnt = main_thread(option.limit, option.nthreads)
        print "number of primes :", cnt
        cnt = main_smp(option.limit, option.nthreads)
        print "number of primes :", cnt
        cnt = main_smp_shared(option.limit, option.nthreads)
        print "number of primes :", cnt
        cnt = main_smp_shared_2(option.limit, option.nthreads)
        print "number of primes :", cnt

def name_tag() :
    return "_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())

def fn_name(routine, tag) :
    return str(routine) + tag + ".data"

def open_file(option_flg, name) :
    f   = sys.stdout
    if option_flg :
        f = open(name, "w")
        print "data written to : " + str(os.getcwd()) +"/"+str(name)
    return f

def close_file(option_flg, f) :
    if option_flg :
        f.close 
    return f

def timing_test(option) :
    tag      = name_tag()
    
    if option.clobber :
        tag      = ""

    routines = map(lambda s : "main_" + s.strip(), option.routines.split(","))
    (fr, to)   = map(lambda i : int(i), option.nthread_range.split(","))
    to += 1
    for routine in routines :
        f = open_file(option.out, fn_name(routine, tag))
        f.write(time_routine_header())
        for nthreads in range(fr, to) :
            args = (option.limit, nthreads, option.repeat)
            l    = repr_tuple("        ")(time_routine(routine,   args))
            f.write(l + "\n")
        #
        f = close_file(option.out, f)

def main(argv) :
    option = dyn_options.create_option(argv, defaults())
    print option 
    if option.help :
        help()
        return 0

    args = (option.limit, option.nthreads, option.repeat)

    if option.nthread_range :
        timing_test(option)
        return

    if option.all or "orig" in option.routines :
        if option.time_it :
            print time_routine("main_orig",   args)
        else :
            cnt = main_orig(option.limit, option.nthreads)
            print "main_orig : ", cnt , " primes found"

    if option.all or "thread" in option.routines :
        if option.time_it :
            print time_routine("main_thread",   args)
        else :
            cnt = main_thread(option.limit, option.nthreads)
            print "main_thread : ", cnt , " primes found"

    if option.all or "smp" in option.routines :
        if option.time_it :
            print time_routine("main_smp",   args)
        else :
            cnt = main_smp(option.limit, option.nthreads)
            print "main_smp : ", cnt , " primes found"


    if option.all or "smp_shared" in option.routines :
        if option.time_it :
            print time_routine("main_smp_shared",   args)
        else :
            cnt = main_smp_shared(option.limit, option.nthreads)
            print "main_smp_shared : ", cnt , " primes found"

        
    if option.all or "smp_shared_2" in option.routines :
        if option.time_it :
            print time_routine("main_smp_shared_2",   args)
        else :
            cnt = main_smp_shared_2(option.limit, option.nthreads)
            print "main_smp_shared_2 : ", cnt , " primes found"

    sys.exit(0) 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 
