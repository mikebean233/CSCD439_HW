inputFile = testfile2
outputFile1 = sortedWord.txt
outputFile2 = sortedOccur.txt

all: hw1
	./hw1 $(inputFile) $(outputFile1) $(outputFile2)

hw1: main.o linkedList.o
	gcc -g -o hw1 main.c linkedList.o

linkedList.o: linkedList.c
	gcc -g -o linkedList.o -c linkedList.c

clean:
	rm *.o hw1
