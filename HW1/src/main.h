#include "linkedList.h"


#define COMPARE_WORD_BY_VALUE 0
#define COMPARE_WORD_BY_COUNT 1

typedef struct{
    int longestLineLength;
    int noLines;
} fileInfo;

typedef struct{
    char* value;
    int count;
} word;

int printError(char* message);
int usage(char* programName, int exitStatus);
LinkedList* tokenize(char* input, int* tokenCount);
int stripNewLine(char* input, int length);
fileInfo* getFileInfo(FILE* filePointer);
int compareWords(Node* nodeA, Node* nodeB, int compareBy);
int strContains(char target, char* source);
char* mikecopy(char* input, int length);
void printWordNode(Node* thisNode);