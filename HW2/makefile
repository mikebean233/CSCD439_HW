All: pgmUtility.c pgmProcessSequential.o pgmProcess.o timing.o
	g++ -g -o myPaintSeq pgmUtility.c pgmProcessSequential.o timing.o -lm
	nvcc -arch=sm_32 -o myPaint pgmUtility.c pgmProcess.o timing.o -lm

pgmProcess.o: pgmProcess.cu
        nvcc -arch=sm_32 -o pgmProcess.o prmProcess.cu

pgmProcessSequential.o: pgmProcessSequential.c
	g++ -g -c -o pgmProcessSequential.o pgmProcessSequential.c

timing.o: timing.c
	g++ -g -c -o timing.o timing.c

clean: *.o
	rm *.o
	rm myPaint*