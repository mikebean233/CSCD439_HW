#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
//#include <math.h>
#include "main.h"
#include "linkedList.h"

int globalCompareFlags;

int main(int argc, char** args)
{
    if(argc != 4)
        usage(args[0], 1);

    // Open the input / output files
    FILE* inputFile       = fopen(args[1], "r");
    FILE* wordOutputFile  = fopen(args[2], "w");
    FILE* occurOutputFile = fopen(args[3], "w");

    if(inputFile == NULL || wordOutputFile == NULL || occurOutputFile == NULL){
        printError("File IO error");

        // Close the files that we managed to open
        if(!(inputFile == NULL))
            fclose(inputFile);
        if(!(wordOutputFile == NULL))
            fclose(wordOutputFile);
        if(!(occurOutputFile == NULL))
            fclose(occurOutputFile);

        exit(2);
    }
    printf("Getting file attributes (line count and longest line length) ...");
    fileInfo* thisFileInfo = getFileInfo(inputFile);
    printf(" Done\n");

    char* thisLine = calloc(sizeof(int), thisFileInfo->longestLineLength + 1);
    LinkedList* tokenList = linkedList();

    // Set the linked list function pointers
    tokenList->compareNodes = &compareWordNodes;
    tokenList->printNode    = &printWordNode;
    tokenList->duplicateEntryBehavior = &duplicateWordBehavior;
    tokenList->nodeToString = &nodeToString;
    tokenList->freeNodeData = &freeWordNode;
    tokenList->compareFlag  = COMPARE_WORD_BY_VALUE;

    int lineCount = 0;
    printf("Tokenizing input file %s into a linked list ...", args[1]);
    while(fgets(thisLine, thisFileInfo->longestLineLength + 1, inputFile) != NULL) {
        stripNewLine(thisLine, thisFileInfo->longestLineLength);

        int noTokens = 0;
        if(strlen(thisLine) > 0){
            tokenize(thisLine, &noTokens, tokenList);

        }
    }
    printf(" Done\n");

    // get the array of word pointers from the linked list
    printf("Converting linked list into array ...");
    word** wordArray = (word**)toArray(tokenList);
    printf(" Done\n");
    //printWordPointerArray(wordArray, tokenList->size);

    // Use qsort to sort by occurrence in descending order
    printf("Sorting words by occurence ...");
    globalCompareFlags = COMPARE_WORD_BY_COUNT | COMPARE_DESC;
    qsort(wordArray, tokenList->size, sizeof(word*), &qsortCompare);
    printf(" Done\n");

    // print the output file
    printf("Writing results to %s ...", args[2]);
    printWordPointerArray(wordArray, tokenList->size, occurOutputFile);
    printf(" Done\n");

    // Use qsort to soft by word value in ascending order
    printf("Sorting by word ...");
    globalCompareFlags = COMPARE_WORD_BY_VALUE | COMPARE_ASC;
    qsort(wordArray, tokenList->size, sizeof(word*), &qsortCompare);
    printf(" Done\n");

    // print to the output file
    printf("Writing results to %s", args[3]);
    printWordPointerArray(wordArray, tokenList->size, wordOutputFile);
    printf(" Done\n");
    //printWordPointerArray(wordArray, tokenList->size);

    // free the resources we've used
    printf("Cleaning up ...");
    free(thisLine);
    fclose(inputFile);
    fclose(wordOutputFile);
    fclose(occurOutputFile);
    clearList(tokenList);
    printf(" Done\n");
    return 0;
 }

/***************************************************************
 *
 *                 LINKED LIST FUNCTIONS
 *
 ***************************************************************/

// LinkedList.printNode()
void printNode(Node* inNode){
    printf("%s", (char*)inNode->data);
}
// LinkedList.dublicateEntryBehavior
int duplicateWordBehavior(struct linkedlist* theList, Node* newNode, Node* match){
    ((word*)(match->data))->count++;
    free(newNode);
    return 0;
}

// LinkedList.freeNodeData()
int freeWordNode(Node* node){
    // free the string
    free( ((word*) node->data)->value );

    // free free the word object
    free( node->data );

    return 1;
}

// LinkedList.nodeToString()
char* nodeToString(Node* node){
    if(node == NULL || node->data == NULL)
        return NULL;

    return (char*)((word*)node->data)->value;
}

// LinkedList.printNode()
void printWordNode(Node* thisNode){
    if(thisNode->data != NULL){
        word thisWord = *((word*)thisNode->data);
        printWord(&thisWord, (FILE*) stdout);
    }
    else
        printError("main.printWordNode received a empty node...");
}
// LinkedList.compareNodes()
int compareWordNodes(Node* nodeA, Node* nodeB, int compareFlags){
    word* wordA = ((word*)nodeA->data);
    word* wordB = ((word*)nodeB->data);
    return compareWords(wordA, wordB, compareFlags);
}

/***************************************************************
 *
 *                 COMPARE FUNCTIONS
 *
 ***************************************************************/

int qsortCompare(const void* worda, const void* wordb){
    return compareWords(*(word**)worda, *(word**)wordb, globalCompareFlags);
}


int compareWords(word* worda, word* wordb, int compareFlags){
    int returnValue = 0;

    if(compareFlags & COMPARE_WORD_BY_VALUE)
        returnValue =  compareStringsIgnoreCase((char*)worda->value, (char*)wordb->value);
    else if(compareFlags & COMPARE_WORD_BY_COUNT)
        returnValue =  worda->count - wordb->count;

    if(compareFlags & COMPARE_DESC)
        returnValue *= -1;

    if(DEBUG){
        char* compareSymbols = "<=>";
        char compareSymbol;

        if(returnValue < 0)
            compareSymbol = compareSymbols[0];
        else if(returnValue == 0)
            compareSymbol = compareSymbols[1];
        else if(returnValue > 0)
            compareSymbol = compareSymbols[2];

        if(compareFlags == COMPARE_WORD_BY_VALUE)
            printf("%-20s %c %-20s\n", worda->value, compareSymbol, wordb->value);
        else if(compareFlags == COMPARE_WORD_BY_COUNT)
            printf("%-20d %c %-20d\n", worda->count, compareSymbol, wordb->count);

    }
    return returnValue;
}


/***************************************************************
 *
 *                   UTILITIES
 *
 ***************************************************************/


void printWordPointerArray(word** array, int size, FILE* destination){
    int i = 0;
    fprintf(destination, "|--------------------|---------|\n");
    fprintf(destination, "|English Word        | Count   |\n");
    fprintf(destination, "|--------------------|---------|\n");
    for(i = 0; i < size; ++i){
        printWord(array[i], destination);
        fprintf(destination, "|--------------------|---------|\n");
    }
}

void printWord(word* thisWord, FILE* destination){
    fprintf(destination, "|%-20s|   %-4d  |\n", thisWord->value, thisWord->count);
}

LinkedList* tokenize(char* input, int* tokenCount, LinkedList* tokenList){

    int length = strlen(input);
    char leadChar, followChar;
    int currentTokenLength = 0;

    int i;
    for(i = 0; i <= length; ++i){
        leadChar   = input[i + 1];
        followChar = input[i];

        int leadOnToken   = isalpha((int)leadChar);
        int followOnToken = isalpha((int)followChar);

        // we are just stepping onto a token
        if(!followOnToken && leadOnToken){
            // do nothing
        }

        // we are in the middle of a token
        if(followOnToken && leadOnToken){
            ++currentTokenLength;
        }

        // we are stepping off of a token
        if(followOnToken && !leadOnToken){
            ++currentTokenLength;
            char* tokenStartPos = input + (i - (currentTokenLength - 1));
            char* thisToken = mikecopy(tokenStartPos, currentTokenLength);
            if(currentTokenLength == 1 && (tolower((int)thisToken[0]) != 'a' && tolower((int)thisToken[0]) != 'i')) {
                currentTokenLength = 0;
                continue;
            }
            word* thisWord = malloc(sizeof(word));
            thisWord->value = thisToken;
            thisWord->count = 1;
            addOrdered(tokenList, thisWord);
            currentTokenLength = 0;
        }

        // we are not touching a token
        if(!followOnToken && !leadOnToken){
        }
    }
    return tokenList;
}

char* mikecopy(char* input, int length){
    char* output = malloc(sizeof(char) * (length + 1));
    int i = 0;
    for(; i < length; ++i){
        output[i] = input[i];
    }
    output[i] = '\0';
    return output;
}

fileInfo* getFileInfo(FILE* filePointer){
    int longestLength = 0;
    int lineCount = 0;
    char thisChar;
    char prevChar = '\0';
    int thisLineLength = 0;
    while((thisChar = fgetc(filePointer)) != EOF){
        if(
                (char)thisChar != '\n' &&
                (char)thisChar != '\r' &&
                (char)prevChar != '\n' &&
                (char)prevChar != '\r')
            ++thisLineLength;
        else{
            ++lineCount;
            if(thisLineLength > longestLength)
                longestLength = thisLineLength;
            thisLineLength = 0;
        }
        prevChar = thisChar;
    }
    rewind(filePointer);

    fileInfo* thisFileInfo = malloc(sizeof(fileInfo));
    thisFileInfo->longestLineLength = longestLength + 1;
    thisFileInfo->noLines = lineCount;
    return thisFileInfo;
}

int printError(char* message){
    fprintf(stderr, "%s : %s", message, strerror(errno));
    return 0;
}

void usage(char* programName, int exitStatus){
    fprintf(stderr, "usage: %s inputFilename wordSortedOutputFile occurrenceSortedOutputFile\n", programName);
    exit(exitStatus);
}

/***************************************************************
 *
 *                    STRING UTILITIES
 *
 ***************************************************************/


int stripNewLine(char* input, int length){
    int index = 0;
    int matchCount = 0;
    for(index = 0; index < length; ++index){
        if(input[index] == '\n' || input[index] == '\r'){
            ++matchCount;
            input[index] = '\0';
        }
    }
    return matchCount;
}

int compareStringsIgnoreCase(char* a, char* b){
    char* newa = calloc(strlen(a), 1);
    char* newb = calloc(strlen(b), 1);

    int i;
    for(i = 0; i < strlen(a); ++i){
        newa[i] = tolower(a[i]);
    }

    for(i = 0; i < strlen(b); ++i){
        newb[i] = tolower(b[i]);
    }

    int returnValue = strcmp(newa, newb);
    free(newa);
    free(newb);
    return returnValue;
}

int strContains(char target, char* source){
    int index = 0;
    while(source[index] != '\0')
        if(source[index++] == target)
            return 1; // true

    return 0; // false
}

