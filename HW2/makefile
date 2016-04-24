All: pgmUtility.c pgmProcessSequential.o pgmProcess.o
	g++ -g -o myPaintSeq pgmUtility.o pgmProcessSequential.o
	nvcc -o myPaint pgmUtility.o pgmProcess.o

pgmProcess.o: pgmProcess.cu
	nvcc -g -c -o pgmProcess.o pgmProcess.c

pgmProcessSequential.o: pgmProcessSequential.c
	g++ -g -c -o pgmProcessSequential.o pgmProcessSequential.c

timing.o: timing.c
	g++ -g -c -o timing.o timing.c