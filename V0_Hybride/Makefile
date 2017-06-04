target: decide4


CFLAGS=-g -fopenmp -O3 -std=c99

main4.o: main4.c projet.h
	mpicc $(CFLAGS) -c main4.c

aux.o: aux.c projet.h
	mpicc $(CFLAGS) -c aux.c

decide4: main4.o aux.o
	mpicc $(CFLAGS) -o $@ $^

.PHONY: clean

exec : 
	mpirun -x OMP_NUM_THREADS=2 -n 4 --hostfile hostfile --bynode ./decide4 "4k//4K//4P w"
clean:
	rm -f *.o decide4
