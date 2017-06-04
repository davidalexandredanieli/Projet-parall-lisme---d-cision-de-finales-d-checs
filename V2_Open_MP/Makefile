target: decide3

CFLAGS=-g -fopenmp -O3 -std=c99

main3.o: main3.c projet.h
aux.o: aux.c projet.h


decide3: main3.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean

exec:
	export OMP_NUM_THREADS=4 && ./decide3 "4k///4K///4P w"

clean:
	rm -f *.o decide3
