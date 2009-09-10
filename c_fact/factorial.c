#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>

void testit(int val)
{
      if (val) {
            fprintf(stderr, "test succeeded..\n");
            return;
      }
      
      fprintf(stderr, "test failed..\n");
}


#define TESTIT(__e, __t)                            \
      fprintf(stderr, "test : expected %ld ; observed %lu \n", __e, __t);   \
      testit(__t == __e); \

size_t factorial(size_t val) 
{
      if (val == 0) return 1;
      return val * factorial(val - 1);
}

size_t factorial2(size_t val, size_t accum) 
{
      if (val == 1) return accum;
      return factorial2(val - 1, val * accum);
}


typedef size_t (Func)(void* , size_t, size_t);


size_t factorial3(void* f, size_t val, size_t acc) {
      if (val == 1) return acc; 
      Func* fptr = (Func * ) f;
      return fptr((void *) f, val-1, acc*val);
}

int main(int argc, char** argv)
{
      size_t f = 1;
      size_t exp = 1;
      size_t accum = 1;
      size_t index = 0;
      if (argc == 2) {
            f = atoi(argv[1]);
      }
      if (f == 0) return 0;

      exp = 1;
      fprintf(stderr, "ulong : %lu\n", ULONG_MAX-1);
      for (index = f; index > 0; index--) {
            if (exp > ULONG_MAX/index) {
                  fprintf(stderr, "overflow detected for %ld; exiting ..\n", f);
                  return 1;
            }
            exp *= index;

      }
      fprintf(stderr, "diff : %lu - ratio %lu\n", 
	      ULONG_MAX - exp, ULONG_MAX/exp);
      TESTIT(exp, factorial(f));
      TESTIT(exp, factorial2(f, accum));
      TESTIT(exp, factorial3((void *) factorial3, f, accum));
      return 0;

}
