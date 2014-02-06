
#source files
SRC = BsplineAnalyticSmoother.cpp

OBJ = $(SRC:.cpp=.o)

PHYLIB = libspclean.so

#include directories
INCLUDES = -I. -I$(MKLROOT)/include

#C++ compiler flags
CFLAGS = -O3 -fPIC -openmp

#compiler
CC = icpc

#library paths
LIBS = -L$(MKLROOT)/lib/intel64/

default: $(PHYLIB)

.cpp.o:
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

#create library
$(PHYLIB):$(OBJ)
	$(CC) -shared  -o $(PHYLIB) $(OBJ) $(LIBS) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lmkl_avx -lmkl_def

	
clean:
	rm -f *.o $(PHYLIB)
