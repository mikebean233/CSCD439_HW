run: hw1
	./hw1 testfile1 sortedWord.txt sortedOccur.txt

hw1: main.o linkedList.o
	gcc -o hw1 main.0 linkedList.o
	
main.o:
	gcc -o main.o -c main.c

linkedList.o: linkedList.c
	gcc -o linkedList.o -c linkedList.c

clean:
	rm *.o hw1
