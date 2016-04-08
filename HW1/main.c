#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include "main.h"

int main(int argc, char** args)
{
    if(argc != 2)
        usage(args[0], 1);

    // Open the input file
    FILE* inputFilePointer;
    inputFilePointer = fopen(args[1], "r");

    if(inputFilePointer == NULL){
        printError("error opening input file");
        exit(2);
    }

    fileInfo* thisFileInfo = getFileInfo(inputFilePointer);

    char* thisLine = calloc(sizeof(int), thisFileInfo->longestLineLength + 1);

    while(fgets(thisLine, thisFileInfo->longestLineLength, inputFilePointer) != NULL) {
            stripNewLine(thisLine, thisFileInfo->longestLineLength);

            int noTokens = 0;
            if(strlen(thisLine) > 0){
                char** theseTokens =  tokenize(thisLine, &noTokens);

            int tokenIndex = 0;
            while(tokenIndex < noTokens)
                printf("[%s]", theseTokens[tokenIndex++]);

            printf("\n");
        }
    }

    free(thisLine);
    fclose(inputFilePointer);
}


int compareWords(word* wordA, word* wordB, int compareBy){
    if(compareBy == COMPARE_WORD_BY_VALUE)
        return strcmp((const char*)wordA->value, (const char*)wordB->value);

    if(compareBy == COMPARE_WORD_BY_COUNT)
        return wordA->count - wordB->count;

    fprintf(stderr, "Error: invalid compareBy parameter %d\n", compareBy);
    return 0;
}

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


char** tokenize(char* input, int* tokenCount){

    (*tokenCount) = 0;
    return (char**)0x0;
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
    thisFileInfo->longestLineLength = longestLength;
    thisFileInfo->noLines = lineCount;
    return thisFileInfo;
}

int printError(char* message){
    fprintf(stderr, "%s : %s", message, strerror(errno));
    return 0;
}

int usage(char* programName, int exitStatus){
    fprintf(stderr, "usage: %s inputFilename\n", programName);
    exit(exitStatus);
    return 0;
}

int strContains(char target, char* source){
    int index = 0;
    while(source[index] != '\0')
        if(source[index++] == target)
            return 1; // true

    return 0; // false
}

