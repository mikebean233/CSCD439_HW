All: pgmUtility.cpp pgmProcessSequential.o pgmProcess.o timing.o
	g++ -g -o myPaintSeq pgmUtility.cpp pgmProcessSequential.o timing.o -lm
	nvcc -arch=sm_32 -o myPaint pgmUtility.cpp pgmProcess.o timing.o -lm

pgmProcess.o: pgmProcess.cu
	nvcc -arch=sm_32 -c -o pgmProcess.o pgmProcess.cu

pgmProcessSequential.o: pgmProcessSequential.cpp
	g++ -g -c -o pgmProcessSequential.o pgmProcessSequential.cpp

timing.o: timing.cpp
	g++ -g -c -o timing.o timing.cpp

clean: *.o
	rm *.o
	rm myPaint*