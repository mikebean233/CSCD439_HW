/**
 *  Michael Peterson
 *  00719713
 *  
 *  CSCD240 Homework 3,4
 *  11-17-2014
 *
 * ---- NOTE ----
 * I have implemented the extra credit by providing a -ce and -c -e option, this is reflected
 * by the output of the usage() function
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include "pgmUtility.h"
#include "timing.h"

#define PARSE_HEADER_ERROR      5
#define PARSE_PIXEL_DATA_ERROR  6
#define ALPHA                   7
#define INTEGER                 8
#define DECIMAL                 9
#define UNKNOWN                 10
#define NULL_TOKEN              11
#define NO_DEST                 12

#define DEBUG                   0

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: You can NOT change the input, output, and argument type of the functions in pgmUtility.h
// NOTE: You can NOT change the prototype (signature) of any functions listed in pgmUtility.h

int countTokens(char *input);
int getToken(int index, char *dest, char *source);
char *getTokenAddress(int index, char *source, int *size);
void printError();
void removeLineFeed(char *str);
void free2dArray(void **array, int noRows);
void usage();
int getType(char *array, int size);
int * linearizeBitmap(int **twoDBitmap, int numRows, int numCols);

char *programName;
int errorFlag; // This is used to keep track of problems that arise inside of functions that do not return specific error details

int main(int argc, char **argv) {
    int drawCircle = 0; // false
    int drawEdge = 0; // false
    int drawLine = 0;
    int makeTest = 0;
    int errorFlag = 0;
    int needInputFile = 1;
    int i = 0;
    int circleCenterRow = 0;
    int circleCenterCol = 0;
    int edgeWidth = 0;
    int radius = 0;
    int points[2][2] = {{0, 0},{0, 0}};
    double testImagePeriod[] = {0.0,0.0};
    char *inputFilename = NULL;
    char *outputFilename = NULL;
    programName = argv[0];

    int **bitmap = NULL;
    int* bitmap1d = NULL;
    char **header = NULL;
    int numRows = 0;
    int numCols = 0;

    FILE *inf = NULL;
    FILE *outf = NULL;

    /**
     * Parse/interpret the command line arguments
     */

    if (argc < 5) {
        usage();
        return 1;
    }

    if (strcmp(argv[1], "-l") == 0) // only draw a line
    {
        if (argc == 8
            && getType(argv[2], strlen(argv[2])) == INTEGER // p1 row
            && getType(argv[3], strlen(argv[3])) == INTEGER // p1 column
            && getType(argv[4], strlen(argv[4])) == INTEGER // p2 row
            && getType(argv[5], strlen(argv[5])) == INTEGER // p2 column
            && argv[6][0] != '-' && getType(argv[6], strlen(argv[6])) != INTEGER  // oldImageFile
            && argv[7][0] != '-' && getType(argv[7], strlen(argv[7])) != INTEGER) // newImageFile
        {
            drawLine = 1;
            sscanf(argv[2], "%i", &(points[0][0]));
            sscanf(argv[3], "%i", &(points[0][1]));
            sscanf(argv[4], "%i", &(points[1][0]));
            sscanf(argv[5], "%i", &(points[1][1]));
            inputFilename = argv[6];
            outputFilename = argv[7];
        }
        else {
            usage();
            return 1;
        }
    }
    else if (strcmp(argv[1], "-e") == 0) // only draw an edge
    {
        if (argc == 5
            && getType(argv[2], strlen(argv[2])) == INTEGER // edgeWidth
            && argv[3][0] != '-' && getType(argv[3], strlen(argv[3])) != INTEGER  // oldImageFile
            && argv[4][0] != '-' && getType(argv[4], strlen(argv[4])) != INTEGER) // newImageFile
        {
            drawEdge = 1;
            sscanf(argv[2], "%i", &edgeWidth);
            inputFilename = argv[3];
            outputFilename = argv[4];
        }
        else {
            usage();
            return 1;
        }

    }
    else if (strcmp(argv[1], "-c") == 0 && strcmp(argv[2], "-e") != 0) // only draw a circle
    {
        if (argc == 7
            && getType(argv[2], strlen(argv[2])) == INTEGER // circleCenterRow
            && getType(argv[3], strlen(argv[3])) == INTEGER // circleCenterCol
            && getType(argv[4], strlen(argv[4])) == INTEGER // radius
            && argv[5][0] != '-' && getType(argv[5], strlen(argv[5])) != INTEGER   // oldImageFile
            && argv[6][0] != '-' && getType(argv[6], strlen(argv[6])) != INTEGER)   // newImageFile
        {
            drawCircle = 1;
            sscanf(argv[2], "%i", &circleCenterRow);
            sscanf(argv[3], "%i", &circleCenterCol);
            sscanf(argv[4], "%i", &radius);
            inputFilename = argv[5];
            outputFilename = argv[6];
        }
        else {
            usage();
            return 1;
        }
    }
    else if (strcmp(argv[1], "-ce") == 0) // draw a circle and edge
    {
        if (argc == 8
            && getType(argv[2], strlen(argv[2])) == INTEGER // circleCenterRow
            && getType(argv[3], strlen(argv[3])) == INTEGER // circleCenterCol
            && getType(argv[4], strlen(argv[4])) == INTEGER // radius
            && getType(argv[5], strlen(argv[5])) == INTEGER // edgeWidth
            && argv[6][0] != '-' && getType(argv[6], strlen(argv[6])) != INTEGER  // oldImageFile
            && argv[7][0] != '-' && getType(argv[7], strlen(argv[7])) != INTEGER) // newImageFile
        {
            drawCircle = 1;
            drawEdge = 1;
            sscanf(argv[2], "%i", &circleCenterRow);
            sscanf(argv[3], "%i", &circleCenterCol);
            sscanf(argv[4], "%i", &radius);
            sscanf(argv[5], "%i", &edgeWidth);
            inputFilename = argv[6];
            outputFilename = argv[7];
        }
        else {
            usage();
            return 1;
        }
    }
    else if (strcmp(argv[1], "-c") == 0 && strcmp(argv[2], "-e") == 0) // draw a circle and edge
    {
        if (argc == 9
            && getType(argv[3], strlen(argv[3])) == INTEGER // circleCenterRow
            && getType(argv[4], strlen(argv[4])) == INTEGER // circleCenterCol
            && getType(argv[5], strlen(argv[5])) == INTEGER // radius
            && getType(argv[6], strlen(argv[6])) == INTEGER // edgeWidth
            && argv[7][0] != '-' && getType(argv[7], strlen(argv[7])) != INTEGER // oldImageFile
            && argv[8][0] != '-' && getType(argv[8], strlen(argv[8])) != INTEGER) // newImageFile
        {
            drawCircle = 1;
            drawEdge = 1;
            sscanf(argv[3], "%i", &circleCenterRow);
            sscanf(argv[4], "%i", &circleCenterCol);
            sscanf(argv[5], "%i", &radius);
            sscanf(argv[6], "%i", &edgeWidth);
            inputFilename = argv[7];
            outputFilename = argv[8];
        }
        else {
            usage();
            return 1;
        }
    }
    else if(strcmp(argv[1], "-test") == 0)
    {
        if(argc == 7
            && getType(argv[2], strlen(argv[2])) == INTEGER // output row count
            && getType(argv[3], strlen(argv[3])) == INTEGER // output column count
            && getType(argv[4], strlen(argv[4])) == DECIMAL // output y period
            && getType(argv[5], strlen(argv[5])) == DECIMAL // output x period
            && getType(argv[6], strlen(argv[6])) == UNKNOWN)// output filename
        {
            numRows = atoi(argv[2]);
            numCols = atoi(argv[3]);
            testImagePeriod[0] = atof(argv[4]);
            testImagePeriod[1] = atof(argv[5]);
            outputFilename = argv[6];
            needInputFile = 0;
            makeTest = 1;
        }
        else
        {
            usage();
            return 1;
        }
    }
    else // invalid input, print usage and exit
    {
        usage();
        return 1;
    }

    /*
     * Open the image file, perform the desired modifications the the data, then save the updated image to the specified output file
     */


    header = (char **) malloc(rowsInHeader * sizeof(char *));
    if(header == NULL){
        fprintf(stderr, "Error: There was a problem allocating memory using malloc()\n");
        exit(1);
    }
    if(needInputFile) {
        if ((inf = fopen(inputFilename, "r")) == NULL) {
            fprintf(stderr, "there was a problem opening %s for input\n\n", inputFilename);
            free(header);
            return 1;
        }

        bitmap = pgmRead(header, &numRows, &numCols, inf); // parse the data in the file
        bitmap1d = linearizeBitmap(bitmap, numRows, numCols);

        fclose(inf);// close the file since we are done with it
    }
    else
    {
        // allocate memory for the bitmap
        bitmap1d = (int*) calloc(sizeof(int), numRows * numCols);
        if(bitmap1d ==  NULL){
            fprintf(stderr, "Error: there was a problem allocating memory using malloc()\n");
        }

        // allocate memory for our header
        for(i = 0; i < rowsInHeader; ++i){
            header[i] = (char*) calloc(sizeof(char), maxSizeHeadRow);
            if(header[i] == NULL){
                --i;
                do{
                    free(header[i]);
                } while(i-- > 0);
                free(header);
                free(bitmap1d);
                fprintf(stderr, "Error: There was a problem allocating memory using malloc()\n");
                exit(1);
            }
        }

        // populate the header
        sprintf(header[0], "P2");
        sprintf(header[1], "# HW 2 test file (Michael Peterson)");
        sprintf(header[2], "%d %d", numCols, numRows);
        sprintf(header[3], "%d", 255);
    }


    if (errorFlag != 0 || bitmap1d == NULL) {
        fprintf(stderr, "There was a problem in pgmRead()\n\n");
        printError();
        free2dArray((void **) header, rowsInHeader);
        free2dArray((void **) bitmap, numRows);
        return 1;
    }

    if ((outf = fopen(outputFilename, "w")) == NULL) {
        fprintf(stderr, "There was a problem opening %s for output \n\n", outputFilename);
        free2dArray((void **) header, rowsInHeader);
        return 1;
    }

    double before = currentTime();
    if (drawCircle)
        pgmDrawCircle(bitmap1d, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);

    if (drawEdge)
        pgmDrawEdge(bitmap1d, numRows, numCols, edgeWidth, header);

    if (drawLine)
        pgmDrawLine(bitmap1d, numRows, numCols, header, points[0][0], points[0][1], points[1][0], points[1][1]);

    if(makeTest)
        buildTestOutput(header, bitmap1d, numRows, numCols, testImagePeriod);
    double after = currentTime();

    printf("time cost: %lf seconds\n", after - before);

    if (pgmWrite(header, bitmap1d, numRows, numCols, outf) == -1) {
        fprintf(stderr, "There was a problem writing to %s\n\n", outputFilename);
    }

    fclose(outf); // close the output file

    // Free up the memory we were using before we exit
    free(bitmap1d);
    free2dArray((void **) header, rowsInHeader);
    free2dArray((void **) bitmap, numRows);
    return 0;

}// end main

void usage() {
    fprintf(stderr, "Usage: %s\n", programName);
    fprintf(stderr, " -e edgeWidth oldImageFile newImageFile\n");
    fprintf(stderr, " -c circleCenterRow circleCenterCol radius oldImageFile newImageFile\n");
    fprintf(stderr, " -ce circleCenterRow circleCenterCol radius edgeWidth oldImageFile newImageFile\n");
    fprintf(stderr, " -c -e circleCenterRow circleCenterCol radius edgeWidth oldImageFile newImageFile\n");
    fprintf(stderr, " -l p1row p1col p2row p2col oldImageFile newImageFile\n");
    fprintf(stderr, " -test noRows noCols yPeriod xPeriod outputFilename\n");

}// end usage

void free2dArray(void **array, int noRows) {
    int i;
    if (array == NULL)
        return;

    for (i = 0; i < noRows; ++i)
        if (array[i] != NULL)
            free(array[i]);

    free(array);
}


void printError() {
    switch (errorFlag) {
        case PARSE_HEADER_ERROR:
            fprintf(stderr, "There was a problem parsing the header.\n");
            break;
        case PARSE_PIXEL_DATA_ERROR:
            fprintf(stderr, "There was a problem parsing the pixel data.");
            break;
    }
}


int **pgmRead(char **header, int *numRows, int *numCols, FILE *in) {
    if (in == NULL) // make sure we actually have a pointer to a file before we do anything else
        return NULL;

    int i = 0, j = 0, tmp = 0; // multi use loop counters
    char temp[10]; //used later to store individual tokens

    for (i = 0; i < rowsInHeader; ++i)
        header[i] = (char *) malloc(maxSizeHeadRow * sizeof(char));

    for (i = 0; i < rowsInHeader; ++i) {
        if (feof(in))// if we're at the end of the file already, we have a problem...
        {
            errorFlag = PARSE_HEADER_ERROR; // set the error flag so that we know where the problem was later
            return NULL;
        }
        fgets(header[i], maxSizeHeadRow, in);
        if(stringLengthIncludeCr(header[i]) == 0){
            errorFlag = PARSE_HEADER_ERROR;
            fprintf(stderr, "prgRead(): There was a problem parsing the input file, the header contains an empty line\n");
            return NULL;
        }
    }

    // remove any new line, or line feed characters from the header lines.
    for (i = 0; i < rowsInHeader; ++i)
        for (j = 0; j < (strlen(header[i])); ++j)
            if (header[i][j] == '\n' || header[i][j] == '\r')
                header[i][j] = '\0';


    // now that we have successfully read in the header information, lets examine it and make sure that it
    // is useful...
    if (strcmp(getTokenAddress(0, header[0], &tmp), "P2") != 0 // make sure the first line only contains "P2"
        || countTokens(header[0]) != 1
        || getTokenAddress(0, header[1], &tmp)[0] != '#'      // make sure the second line starts with a #
        || countTokens(header[2]) != 2                        // make sure the third line contains two integers
        || getToken(0, temp, header[2]) != INTEGER
        || getToken(1, temp, header[2]) != INTEGER
        || countTokens(header[3]) != 1                        // make sure the fourth line contains one integer
        || getToken(0, temp, header[3]) != INTEGER) {
        errorFlag = PARSE_HEADER_ERROR;
        return NULL;
    }

    getToken(0, temp, header[2]);
    sscanf(temp, "%i", numCols);
    getToken(1, temp, header[2]);
    sscanf(temp, "%i", numRows);

    int **bitmap = (int **) malloc(*numRows * sizeof(int *));
    if (bitmap == NULL){ // Make sure there wasn't a problem allocating memory for bitmap
        for(i = 0; i < rowsInHeader; ++i)
            if(header[i] != NULL)
                free(header[i]);
        return NULL;
    }

    for (i = 0; i < *numRows; ++i) {
        bitmap[i] = (int *) malloc(*numCols * sizeof(int));
        if (bitmap[i] == NULL) {
            fprintf(stderr, "There was a problem allocating memory for the bitmap...\n\n");
            return NULL;
        }
    }

    int thisLineLength = longestLineLength(in);
    char* thisLine = (char*) calloc(sizeof(char), thisLineLength);


    int numTokens = 0, type = 0, lineNo = 5;
    int r, c;
    int pixelNo = 0;

    while (!feof(in)) {
        fgets(thisLine, thisLineLength, in);
        removeLineFeed(thisLine);

        numTokens = countTokens(thisLine);
        for (i = 0; i < numTokens; ++i) {
            type = getToken(i, temp, thisLine);

            if (type != INTEGER && type != NULL_TOKEN) {
                fprintf(stderr, "\n We encountered a non integer in the pixel data (%s is type %i)\n", temp, type);
                errorFlag = PARSE_PIXEL_DATA_ERROR;
                free(thisLine);
                return NULL;
            }

            r = pixelNo / (*numCols);
            c = pixelNo - (r * *numCols);
            //printf("\n  row: %i   col: %i  LineNo: %i  pixelNo: %i  totalpixels: %i\n", r, c, lineNo, pixelNo, *numRows * *numCols);

            /*
              if(pixelNo > (*numRows * *numCols))
              {
                  printf("\n We encountered more pixels then we should have in the file...\n");
                  printf("row: %i  col: %i\n", r,c);
                  errorFlag = PARSE_PIXEL_DATA_ERROR;
                  return NULL;
              }
            */

            //if(type != NULL_TOKEN)
            //{

            if (r < *numRows && c < *numCols) {
                //printf("parsing token \"%s\" into bitmap[%i][%i]\n", temp, r,c);
                sscanf(temp, "%i", &bitmap[r][c]);
                ++pixelNo;
            }
            //}
        }
        ++lineNo;
    }

    //if(r < (*numRows - 1) || c < (*numCols -1) )
    //{
    //    printf("There were fewer pixels in the file there there were supposed to be...\n");
    //    printf("Last row: %i  Last Column: %i\n", r, c);
    //    return NULL;
    //}
    free(thisLine);
    return bitmap;
}// end prmRead

int pgmWrite(char **header, int *pixels, int numRows, int numCols, FILE *out) {
    if (out == NULL || header == NULL || pixels == NULL)
        return -1;

    int outCols = 16;
    int i, r, c, colCount = 0;
    for (i = 0; i < rowsInHeader; ++i)
        fprintf(out, "%s\n", header[i]);

    for (r = 0; r < numRows; ++r) {
        for (c = 0; c < numCols; ++c) {
            i = numCols * r + c;
            fprintf(out, "%3i ", pixels[i]);
            ++colCount;
            if (colCount > outCols) {
                fprintf(out, "\n");
                colCount = 0;
            }
        }
    }

    return 0;

}


void removeLineFeed(char *str) {
    if (str == NULL)
        return;

    int noChars = strlen(str);

    int i;
    for (i = 0; i < noChars; ++i) {
        if (str[i] == '\n' || str[i] == '\r')
            str[i] = '\0';
    }

    return;

}


/**
 * This function will take a token from a string, store it in the location provided, and 
 * return a int representing the type of information that was contained in the token.
 *
 * @param index: the index of the desired token
 * @param dest: A pointer to the memory location where the token is to be stored
 * @param source: A pointer to the memory location where the string (containing the token) is located
 *
 * return values: 
 */
int getToken(int index, char *dest, char *source) {
    if (source == NULL || strlen(source) == 0)
        return NULL_TOKEN;

    if (dest == NULL)
        return NO_DEST;

    int size = 0;

    char *tokenAddress = NULL;
    tokenAddress = getTokenAddress(index, source, &size);
    // printf("token: %s  size: %i\n", tokenAddress, size);
    if (tokenAddress == NULL || size == 0)
        return NULL_TOKEN;

    int i;

    char *tempCopy = NULL;

    // allocate memory for tempCopy then make sure theree wasn't a problem doing so
    tempCopy = (char *) malloc((size + 1) * sizeof(char));
    if (tempCopy == NULL)
        return NULL_TOKEN;
    for (i = 0; i < size; ++i)
        tempCopy[i] = ' ';

    for (i = 0; i < size; ++i)
        tempCopy[i] = tokenAddress[i];
    tempCopy[size] = '\0';

    // now copy the token data into the address pointed to by dest
    strcpy(dest, tempCopy);

    free(tempCopy);

    return getType(dest, size);

}// end getToken


int getType(char *str, int size) {

    int digitCount = 0;
    int alphaCount = 0;
    int periodCount = 0;
    int otherCount = 0;
    int minusCount = 0;
    int i;

    // count how many of each catagory of character that we have so that we can determine what kind of data the
    // token contains
    for (i = 0; i < size; ++i) {
        char chr = str[i];
        if (isalpha(chr))
            alphaCount++;
        if (isdigit(chr))
            digitCount++;
        if (chr == '.')
            periodCount++;
        if (chr == '-')
            minusCount++;

        if (!isalpha(chr) && !isdigit(chr) && chr != '.')
            otherCount++;
    }

    // We have only letters
    if (alphaCount == size)
        return ALPHA;

    // we have only numbers
    if (digitCount == size || (digitCount == size - 1 && str[0] == '-'))
        return INTEGER;

    // we have a decimal number
    if (periodCount == 1 && digitCount == size - 1 && digitCount > 0)
        return DECIMAL;

    // we have something else
    return UNKNOWN;
}// end getType

char *getTokenAddress(int index, char *source, int *size) {
    int thisToken = -1;
    char dummyChar = ' ';
    char *prevChar = NULL;
    prevChar = &dummyChar;
    char *curChar = NULL;
    curChar = source;

    // Step through all of the characters in the array until we get to the requested token
    while (*curChar != '\0') {
        // check to see if we just stepped onto a token
        if (isspace(*prevChar) && !isspace(*curChar))
            thisToken++;

        if (thisToken == index)
            break;

        prevChar = curChar;
        curChar += 1;
    }

    *size = 0;
    char *sizeCursor = curChar;

    // determine how many characters long the token was and store the result in *size
    while (!isspace(*sizeCursor) && *sizeCursor != '\0') {
        sizeCursor++;
        (*size)++;
    }

    if (*curChar == '\0')
        return NULL;
    else
        return curChar;
}// getTokenAddress

int countTokens(char *input) {

    if (input == NULL)
        return 0;

    int thisToken = 0;
    char dummyChar = ' ';
    char *prevChar = &dummyChar;
    char *curChar = input;

    // Step through all of the characters in the array
    while (*curChar != '\0') {
        // check to see if we just stepped onto a token
        if (isspace(*prevChar) && !isspace(*curChar))
            thisToken++;

        prevChar = curChar;
        curChar += 1;
    }

    //printf("\"%s\"   tokens: %i\n", input, thisToken);
    return thisToken;
}

int printString(char *str) {
    int i = 0;
    int length = strlen(str);

    for (i = 0; i < length; ++i)
        printf("%c", *(str + i));

    printf("\n");

    for (i = 0; i < length; ++i)
        printf("%p %c\n", (str + i), *(str + i));

    printf("\n");

    return 0;

}

/**
 *  Function Name;
 *      intMax()
 *      intMax() returns the maximum of two signed integers
 */
int intMax(int a, int b) {
    if (a > b)
        return a;
    else;
    return b;
}

/**
 *  Function Name;
 *      intMin()
 *      intMin() returns the minimum of two signed integers
 */
int intMin(int a, int b) {
    if (a < b)
        return a;
    else
        return b;
}

int longestLineLength(FILE* filePointer){
    int longestLength = 0;
    int lineCount = 0;
    char thisChar;
    char prevChar = '\0';
    int thisLineLength = 0;
    fpos_t beforePosition;
    fgetpos(filePointer, &beforePosition);

    while((thisChar = fgetc(filePointer)) != EOF){
        if((char)thisChar != '\n' &&
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
    fsetpos(filePointer, &beforePosition);
    return longestLength + 1;
}

int * linearizeBitmap(int **twoDBitmap, int numRows, int numCols){
    if(twoDBitmap == NULL || numRows < 1 || numCols < 1)
        return NULL;

    int* outputArray = (int*) calloc(sizeof(int), numCols * numRows);
    if(outputArray == NULL)
        return NULL;

    int i, row, col;
    for(i = 0; i < numCols * numRows; ++i){
        row = i / numCols;
        col = i % numCols;
        outputArray[i] = twoDBitmap[row][col];
    }
    return outputArray;
}

void buildTestOutput(char** header, int* pixels, int numRows, int numCols, double* period){
    int i = 0;
    int r, c;
    int pixelCount = numRows * numCols;
    double angleA, angleB;
    double outputA, outputB;
    const double PI = acos(-1.0);

    for(i = 0; i < pixelCount; ++i){
        r = i / numCols;
        c = i % numCols;

        angleA = 2 * PI * period[0] * ((double)r / (double)numRows);
        angleB = 2 * PI * period[1] * ((double)c / (double)numCols);
        outputA = (sin(angleA) / 2.0) + 0.5;
        outputB = (sin(angleB) / 2.0) + 0.5;

        pixels[i] = ((outputA + outputB) / 2) * 255;
        sprintf(header[3], "%i", 255);
    }
}

int stringLengthIncludeCr(char* input){
    int i = 0;
    while(input[i] != '\0' && input[i] != '\r' && input[i] != '\n'){
        ++i;
    }
    return i;
}










