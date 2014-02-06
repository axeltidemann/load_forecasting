CC  = gcc
CXX = g++

OPTS = -O3

.SUFFIXES:

clean:
	rm -f *~ *.o a.out

%.o: %.cpp force
	$(CXX) $(OPTS) $(INCLUDE) -c $<

%: %.cpp force
	$(CXX) $(OPTS) -o $(subst _,-,$@) $<
force: ;
