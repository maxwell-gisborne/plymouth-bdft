hash = $(shell sha256sum lib.c | cut -d' ' -f 1)

lib.so: lib.o
	gcc lib.o -shared -lfftw3 -lm -o lib.so
	# -lasan

lib.o: lib.c makefile
	# turn off function inlineing for debug build
	gcc --std=c23 -g -Dinline="" -D__FILE_HASH__=\"$(hash)\" -c -fPIC lib.c

	# -fsanitize=address 

clean:
	rm lib.o lib.so
