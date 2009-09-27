#!/usr/bin/env python

import sys
import os
import math
import thread
try :
    import multiprocessing
except :
    print "multprocessing module not found !!"

import string
import time

from timeit import Timer

import dyn_options

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

""""
Original version, lifted from ....
Notice that all threads share nexti and prime..

"""
def main_orig(top, nthreads):
    global n,prime_global,nexti_global,nextilock,nstarted,nstartedlock,donelock

    n        = int(top)
    nthreads  = int(nthreads)
    prime_global = (n+1) * [1]

    nstarted = 0
    nexti_global = 2
    
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

    return count_primes(prime_global)    

def dowork_orig(tn): # thread number tn
    global n,prime_global,nexti_global,nextilock,nstarted,nstartedlock,donelock
    donelock[tn].acquire()
    nstartedlock.acquire()
    nstarted += 1
    nstartedlock.release()
    lim = math.sqrt(n)

    while 1:
        nextilock.acquire()
        k = nexti_global
        nexti_global += 1
        nextilock.release()
        if k > lim: break
        if prime_global[k]:
            r = n / k
            for i in range(2,r+1):
                prime_global[i*k] = 0

    donelock[tn].release()


"""
Remove shared variable nexti_global; a little faster !
Uses naive load balancing which works because prime 
is shared..
"""

def main_nolocks(top, nthreads) :
    global prime_nl
    n             = int(top)
    nthreads      = int(nthreads)
    
    prime_nl      = (n + 1)  * [1]
    donelock      = map(lambda l : l.acquire() and l, 
                        map(lambda i : thread.allocate_lock(), range(nthreads)))

    lim   = int(math.sqrt(n)) + 1
    nexti_ns = range(2, lim, 1) 

    B = load_balance(nexti_ns, nthreads)

    map(lambda i : start_th(dowork_th, (i, donelock[i], n, B[i])),
        range(nthreads))
    
    map(lambda i : donelock[i].acquire(), range(nthreads) )

    return count_primes(prime_nl)    

def dowork2(n, nexti_ns, prime_nl) :

    k     = nexti_ns[0]
    lim   = nexti_ns[1]
    if nexti_ns[0] > nexti_ns[1] : 
        raise "boundaries out-of-order"

    while 1 :

        if not (k < lim) : break

        if prime_nl[k] == 1 :
            r = n / k
            for i in range(2, r+1) :
                prime_nl[i*k] = 0

        k   = k + 1


    return prime_nl


def dowork_th(tn, donelock, n, nexti_ns) :
    global prime_nl
    prime_nl = dowork2(n, nexti_ns, prime_nl)
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


"""
Splitting up the seive array between threads
"""
def main_nolocks_alt(top, nthreads) :
    global prime_nla
    n             = int(top)
    nthreads      = int(nthreads)

    prime_nla     = (n + 1)  * [1]
    donelock      = map(lambda l : l.acquire() and l, 
                        map(lambda i : thread.allocate_lock(), range(nthreads)))

    ind   = range(2, n, 1)
    B = load_balance(ind, nthreads)
    print B
    map(lambda i : start_th(dowork_th3, (i, donelock[i], n, B[i])),
        range(nthreads))
    
    map(lambda i : donelock[i].acquire(), range(nthreads) )
        
    return count_primes(prime_nla)    

def dowork3(n, irange, prime_nla) :
    ops   = 0
    lim   = int(math.sqrt(n)) + 1
    istart, iend = irange
    k     = istart
    while 1 :

        if not (k < lim)  : break
        if not (k < iend) : break

        if k < istart :
            s = (istart / k ) + 1
            r = (iend / k) + 1 
            for i in range(s, r) :
                ops += 1
                prime_nla[i*k] = 0
        elif prime_nla[k] == 1 :
            assert k >= istart and k <= iend
            s = 2
            r = (iend / k) + 1
            for i in range(s, r) :
                ops += 1
                prime_nla[i*k] = 0

        k   = k + 1
    print "istart : ", istart, " iend: ", iend, "operations : ", ops
    return prime_nla 

def dowork_th3(tn, donelock, n, rangei) :
    global prime_nla
    prime_nla = dowork3(n, rangei, prime_nla)
    donelock.release()


"""
Load balance the indices. A more equitable work distribution

"""
def main_nolocks_alt2(top, nthreads) :
    global prime_nls4
    n             = int(top)
    nthreads      = int(nthreads)

    prime_nls4    = (n + 1)  * [1]

    donelock      = map(lambda l : l.acquire() and l, 
                        map(lambda i : thread.allocate_lock(), range(nthreads)))

    #print nexti

    B = smp_load_balance(nthreads, n)
    #print B
    map(lambda i : start_th(dowork_th4, (i, donelock[i], n, B[i])),
        range(nthreads))
    
    map(lambda i : donelock[i].acquire(), range(nthreads) )
        
    return count_primes(prime_nls4)    

def dowork4(n, next_arr, prime_nls4) :

    nk    = 0
    lim   = len(next_arr)
    while 1 :
        k  = next_arr[nk]
        if prime_nls4[k] == 1 :
            r = n / k
            for i in range(next_arr[0], r+1) :
                prime_nls4[i*k] = 0
        nk += 1
        if nk >= lim : break

    return prime_nls4


def dowork_th4(tn, donelock, n, nexti) :
    global prime_nls4
    prime = dowork4(n, nexti, prime_nls4)
    donelock.release()



"""
Using process Pools; No shared variables; 
Load balancing tries to keep the number of operations
per process constant

"""
def main_smp(top, nthreads) :

    n             = int(top)
    nthreads      = int(nthreads)

    B     = smp_load_balance(nthreads, n)
    p     = multiprocessing.Pool(nthreads)
    K     = p.map(dowork_smp, map(lambda lst : (n, lst, nthreads), B))
    PR    = transpose(K)

    prime = p.map(reduce_chunk, PR)
    return count_primes(prime)    

def dowork_smp(args) :
    n, nexti_smp, chunks = args
    nk    = 0
    ops   = 0
    k     = nexti_smp[0]
    L     = ( n + 1) * [1]
    lim   = len(nexti_smp)
    while 1 :

        k  = nexti_smp[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti_smp[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

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
    nexti_lb      = range(2, lim, 1)

    if th < 2 :
        return [nexti_lb]

    thr_allocs = map(lambda i : (0, [] ), range(th))
    Z = map(operations, nexti_lb)

    L = zip(map(operations, nexti_lb), nexti_lb)

    for i in L :
        ops, index = i
        mindex = find_min(thr_allocs)
        cnt, lst = thr_allocs[mindex]
        cnt += ops
        lst.append(index)
        thr_allocs[mindex] = (cnt, lst)

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


"""
altermative distribution of work: Split the sieve between the processes

"""
def main_smp_alt(top, nthreads) :

    n             = int(top)
    nthreads      = int(nthreads)

    prime_smp_alt = (n + 1) * [1]

    
    ind   = range(2, n, 1)
    B     = load_balance(ind, nthreads)
    p     = multiprocessing.Pool(nthreads)
    K     = p.map(dowork_smp_alt, map(lambda lst : (n, lst), B))

    prime_smp_alt = reduce(lambda l,r : l + r, K)
    return count_primes(prime_smp_alt)

def dowork_smp_alt(args) :
    n, irange    = args
    k            = 2
    lim          = int(math.sqrt(n)) + 1
    istart, iend = irange
    L            = ( n + 1) * [1]
    ifrom        = 999999
    ito          = -1

    while 1 :

        if not (k < lim)  : break
        if not (k < iend) : break

        if k < istart :
            s = (istart / k ) + 1
            r = (iend / k) + 1 
            for i in range(s, r) :
                index = i * k

                if ifrom > index :
                    ifrom = index
                if ito < index :
                    ito = index

                L[i*k] = 0
        elif L[k] == 1 :
            s = 2
            r = (iend / k) + 1
            for i in range(s, r) :

                index = i * k

                if ifrom > index :
                    ifrom = index
                if ito < index :
                    ito = index

                L[i*k] = 0

        k   = k + 1

    if ifrom == 4 :
        ifrom = 0

    return L[ifrom: ito + 1] 


"""
Introduce shared variable in the smp module...

"""
def main_smp_shared(top, nthreads) :

    n             = int(top)
    nthreads      = int(nthreads)

    manager = multiprocessing.Manager()
    prime_s = (n + 1) * [1] 
    B       = smp_load_balance(nthreads, n)
    p       = multiprocessing.Pool(nthreads)
    L_m     = manager.list(prime_s)
    K       = p.map(dowork_smp_shared, map(lambda lst : (n, lst, L_m), B))

    return count_primes(L_m)    

def dowork_smp_shared(args) :
    n, nexti_shared, L = args
    nk    = 0
    ops   = 0
    k     = nexti_shared[0]
    lim   = len(nexti_shared)
    while 1 :

        k  = nexti_shared[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti_shared[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

    return []



"""
Uses processes, and shared data...

"""

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

def create_process(argv) :
    return multiprocessing.Process(target=dowork_smp_shared_2, args=(argv,))

def dowork_smp_shared_2(args) :
    n, nexti_sh2, L = args
    nk    = 0
    ops   = 0
    k     = nexti_sh2[0]

    lim   = len(nexti_sh2)
    while 1 :

        k  = nexti_sh2[nk]
        if L[k] == 1 :
            r = n / k
            for i in range(nexti_sh2[0], r+1) :
                ops   += 1
                L[i*k] = 0
        nk += 1
        if nk >= lim : break

    return L

    
    
"------------------------------------------------"
def time_routine_header() :
    return "description   nthreads   repeat    limit       time (sec)\n"

def time_routine(desc, args) :

    (limit, nthreads, repeat) = args

    cmd   = "%s(%s, %s) " % (desc, limit, nthreads)
    setup = "from __main__ import %s" % desc 
    t     = Timer(cmd, setup)

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

    def get_range(fr, to) :
        if fr < to :
            to += 1
            return range(fr, to)
        if fr > to :
            to -= 1
            return range(fr, to, -1)
        return range(fr, to)

    tag      = name_tag()
    
    if option.clobber :
        tag      = ""

    routines = map(lambda s : "main_" + s.strip(), option.routines.split(","))
    (fr, to)   = map(lambda i : int(i), option.nthread_range.split(","))

    for routine in routines :
        f = open_file(option.out, fn_name(routine, tag))
        f.write(time_routine_header())
        for nthreads in get_range(fr, to) :
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

    routines = map(lambda s : s.strip(), option.routines.split(","))

    if option.all or "orig" in routines :
        if option.time_it :
            print time_routine("main_orig",   args)
        else :
            cnt = main_orig(option.limit, option.nthreads)
            print "main_orig : ", cnt , " primes found"

    if option.all or "nolocks" in routines :
        if option.time_it :
            print time_routine("main_nolocks",   args)
        else :
            cnt = main_nolocks(option.limit, option.nthreads)
            print "main_nolocks : ", cnt , " primes found"

    if option.all or "nolocks_alt" in routines :
        if option.time_it :
            print time_routine("main_nolocks_alt",   args)
        else :
            cnt = main_nolocks_alt(option.limit, option.nthreads)
            print "main_nolocks_alt : ", cnt , " primes found"

    if option.all or "nolocks_alt2" in routines :
        if option.time_it :
            print time_routine("main_nolocks_alt2",   args)
        else :
            cnt = main_nolocks_alt2(option.limit, option.nthreads)
            print "main_nolocks_alt2 : ", cnt , " primes found"

    if option.all or "smp" in routines :
        if option.time_it :
            print time_routine("main_smp",   args)
        else :
            cnt = main_smp(option.limit, option.nthreads)
            print "main_smp : ", cnt , " primes found"

    if option.all or "smp_alt" in routines :
        if option.time_it :
            print time_routine("main_smp_alt",   args)
        else :
            cnt = main_smp_alt(option.limit, option.nthreads)
            print "main_smp_alt : ", cnt , " primes found"


    if option.all or "smp_shared" in routines :
        if option.time_it :
            print time_routine("main_smp_shared",   args)
        else :
            cnt = main_smp_shared(option.limit, option.nthreads)
            print "main_smp_shared : ", cnt , " primes found"

        
    if option.all or "smp_shared_2" in routines :
        if option.time_it :
            print time_routine("main_smp_shared_2",   args)
        else :
            cnt = main_smp_shared_2(option.limit, option.nthreads)
            print "main_smp_shared_2 : ", cnt , " primes found"

    sys.exit(0) 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 
