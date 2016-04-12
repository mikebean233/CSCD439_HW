#include "linkedList.h"

#define DEBUG 0
#define COMPARE_WORD_BY_VALUE 0x1
#define COMPARE_WORD_BY_COUNT 0x2
#define COMPARE_ASC           0x4
#define COMPARE_DESC          0x8

typedef struct{
    int longestLineLength;
    int noLines;
} fileInfo;

typedef struct{
    char* value;
    int count;
} word;

int printError(char* message);
void usage(char* programName, int exitStatus);
LinkedList* tokenize(char* input, int* tokenCount, LinkedList* tokenList);
int stripNewLine(char* input, int length);
fileInfo* getFileInfo(FILE* filePointer);
int strContains(char target, char* source);
char* mikecopy(char* input, int length);
void printWordNode(Node* thisNode);
void printWord(word* thisWord, FILE* destination);
int stringCompare(Node* inA, Node* inB, int param);
void printNode(Node* inNode);
int duplicateWordBehavior(struct linkedlist* theList, Node* newNode, Node* match);
int compareStringsIgnoreCase(char* a, char* b);
int compareWordNodes(Node* nodeA, Node* nodeB, int compareFlags);
int compareWords(word* worda, word* wordb, int compareFlags);
int qsortCompare(const void* worda, const void* wordb);
char* nodeToString(Node* node);
int freeWordNode(Node* node);
void printWordPointerArray(word** array, int size, FILE* destination);
