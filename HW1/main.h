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
LinkedList* tokenize(char* input, int* tokenCount, LinkedList* tokenList);
int stripNewLine(char* input, int length);
fileInfo* getFileInfo(FILE* filePointer);
int strContains(char target, char* source);
char* mikecopy(char* input, int length);
void printWordNode(Node* thisNode);
int stringCompare(Node* inA, Node* inB, int param);
void printNode(Node* inNode);
int duplicateWordBehavior(struct linkedlist* theList, Node* newNode, Node* match);
int compareStringsIgnoreCase(char* a, char* b);
int compareWordNodes(Node* nodeA, Node* nodeB, int compareBy);
int compareWords(word* worda, word* wordb, int compareBy);
void printWord(word* thisWord);
int qsortCompare(const void* worda, const void* wordb);
