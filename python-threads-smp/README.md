# python-thread-smp

## What is it ?

It's a simple python script used to explore multi threading and multiprocessing
in python.
It test a simple compute-bound algorithm to count the number of primes up to a certain limit.
The basic implementation can be found in chapter five of [this tutorial (pdf)](heather.cs.ucdavis.edu/~matloff/Python/PyThreads.pdf). 

## Usage

### Timing test
    
  The purpose of this script is to be able to run timing tests of the various implementations.

  A timing test is kicked off either by specifying  a range of threads using 
  the -nthread\_range option (see below), or using the -time_it option (see below). 


### Command line options

*prime_combine.py* implements a few variations of the algorithm and takes various command line options.


#### names of the routines
 
  * orig
  * nolocks
  * nolocks_alt
  * nolocks_alt2
  * smp
  * smp_shared
  * smp_shared_2

See [prognotes](www.prognotes.com) for a detailed discussion of these options

#### -routines 'routine1',.... 

  A comma separated list of the routines to run. 

#### -limit 'limit'
  Count all primes up to 'limit'.

#### -repeat 'repeat'
  Repeat the experiment 'repeat' times.

#### Timing Experiments

*prime_combine.py* is designed to easily run timing experiments on the various implementations of
the prime counting algorithm. You have to option to run a set of experiments over a range of routines
and threads, or do just a single experiment.


##### -nthread_range 'from', 'to'
  Run a timing test by varying the number of threads/process from 'from' to 'to', inclusive.
    
##### -time_it 
  Run a single test
   
#### output files
  These options apply only if you chose to run a timing test by specifying the thread range.

##### -out

  Capture the output in a file. The file name is auto-generated and consists of the routine name 
  and a date/time stamp.

##### -clobber
  
  In this case the date/time is omitted from the file name, potentially clobbering previously
  generated data files


Example :


    ./prime_combine.py -repeat 40 -limit 10000 -nthread_range 1,6 -routines orig, thread -clobber -out


## Dependencies

It uses [dyn_options](git://github.com/fons/dyn_options.git) to process the command line arguments.
