INCLUDES=-I ../FunctionalUtilities -I ../AutoDiff
test:test.o
	g++ -std=c++14 -O3 -pthread --coverage test.o $(INCLUDES) -o test -fopenmp
test.o:test.cpp Newton.h
	g++ -std=c++14 -O3 -pthread --coverage -c test.cpp $(INCLUDES) -fopenmp
clean:
	-rm *.o *.out test