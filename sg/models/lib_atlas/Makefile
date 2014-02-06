
# source files
#SRC = BsplineAnalyticSmoother.cpp demoCleansing.cpp
SRC = BsplineAnalyticSmoother.cpp

OBJ = $(SRC:.cpp=.o)

PHYLIB = libspclean.so

# #include directories
#INCLUDES = -I. -I/usr/local/include -I/usr/local/include/atlas -I/usr/include
#INCLUDES = -I. -I/usr/local/include -I/usr/local/include/atlas -I/home/hasib/software/mathlib/plasma-installer_2.5.0b1/install/include -I$(HOME)/include
INCLUDES = -I. -I/usr/local/include -I/usr/include -I/usr/local/include/atlas -I$(HOME)/include

#C++ compiler flags
#CFLAGS = -O3 -fPIC -Wall	#for lib
CFLAGS = -O3 -fPIC -fopenmp -Wall	#for lib
#CFLAGS = -O3 -fopenmp	#for executable
#CFLAGS = -g -pg -O0 -Wall -fPIC # for debugging

#compiler
CC = g++

#library paths
hostname := $(shell hostname)

ifeq ($(hostname), rocks.hpc.ntnu.no)
   LIBS = -llapack -lptcblas -lptf77blas -latlas -lm -lrt -lpthread -lgomp -lgfortran
else
   #LIBS = -L/home/hasib/software/mathlib/plasma-installer_2.5.0b1/install/lib -L/usr/lib64 -L/usr/local/lib64
   LIBS = -L/usr/local/lib64 -L$(HOME)/lib/ptlib -lptlapack -lptcblas -lptf77blas -latlas -lm -lrt -lpthread -lgomp -L/usr/lib64/gcc/x86_64-suse-linux/4.6/ -lgfortran
   #LIBS = -L/usr/local/lib64 -L$(HOME)/lib/ptlib -lptlapack -lptcblas -lptf77blas -latlas -lm -lrt -lpthread -lgomp -L/usr/lib64/gcc/x86_64-suse-linux/4.6/ -lgfortran
endif

default: $(PHYLIB)

.cpp.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(PHYLIB):$(OBJ)
	$(CC) -shared -o $(PHYLIB) $(OBJ) $(LIBS)

#$(CC) -shared -o $(PHYLIB) $(OBJ) $(LIBS) -lplasma -lcoreblas -lquark -lcblas -llapacke -ltmg -llapack -lf77blas -latlas -lm -lrt -lhwloc -lrt -lpthread -lgomp -L/usr/lib64/gcc/x86_64-suse-linux/4.6/ -lgfortran
#$(CC) -shared  -o $(PHYLIB) $(OBJ) $(LIBS) -lplasma -lcblas -lcoreblas -lquark -llapacke -ltmg -llapack -lf77blas -latlas -lm -lrt -lhwloc -lpthread -lgomp
#$(CC) -shared  -o $(PHYLIB) $(OBJ) $(LIBS) -llapack -lptcblas -lptf77blas -latlas -lpthread -lgomp
#$(CC)  -o $(PHYLIB) $(OBJ) $(LIBS) -lplasma -lcoreblas -lquark -lcblas -llapacke -ltmg -llapack -lf77blas -latlas -lm -lrt -lhwloc -lrt -lpthread
#$(CC)  -o $(PHYLIB) $(OBJ) $(LIBS) -lpthread -lptcblas -latlas -llapack -llapacke -lquark -ltmg -lcoreblas  -lplasma  -lm -lrt
#$(CC) -shared -o $(PHYLIB) $(OBJ) $(LIBS) -lcblas -latlas -lgomp

clean:
	rm -f *.o $(PHYLIB)

