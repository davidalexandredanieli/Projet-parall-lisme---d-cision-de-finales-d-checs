target: decide2


CFLAGS=-g -O3 -std=c99
CC=mpicc
main2.o: main2.c projet.h
aux.o: aux.c projet.h

decide2: main2.o aux.o
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY: clean

exec : 
	mpirun -n 4 --hostfile hostfile --bynode ./decide2 "5k//2K1P b"

clean:
	rm -f *.o decide2
