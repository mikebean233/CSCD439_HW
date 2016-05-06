#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "pgmUtility.h"

#define DEBUG 1

// intWithin() flags
#define LOWER_INCLUSIVE         1
#define UPPER_INCLUSIVE         2


double distance(int p1[], int p2[]);
int intInRange(int target, int bound1, int bound2, int flags);


int pgmDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, char **header) {
    if (pixels == NULL || header == NULL)
        return 0;
    int i = 0;
    int oldMaxIntens = 0;
    int newMaxIntens = 0;
    sscanf(header[3], "%i", &oldMaxIntens);

    int r, c;
    for (r = 0; r < numRows; ++r) {
        for (c = 0; c < numCols; ++c) {
            i = numCols * r + c;
            if (c < edgeWidth           // left edge
                || c > numCols - edgeWidth // right edge
                || r < edgeWidth           // top edge
                || r > numRows - edgeWidth) // bottom edge
                pixels[i] = 0;
            if (pixels[i] > newMaxIntens)
                newMaxIntens = pixels[i];
        }
    }

    sprintf(header[3], "%i", newMaxIntens);
    return (newMaxIntens == oldMaxIntens) ? 0 : 1;
}// end pgmDrawEdge


int pgmDrawCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header) {
    if (pixels == NULL || header == NULL)
        return 1;

    int oldMaxIntens = 0;
    int newMaxIntens = 0;
    sscanf(header[3], "%i", &oldMaxIntens);
    radius = abs(radius); // so we don't have to deal with any negative radius nonsense

    int r, c;
    int i;
    int p1[] = {centerRow, centerCol};
    int p2[2];
    for (r = 0; r < numRows; ++r) {
        for (c = 0; c < numCols; ++c) {
            i = numCols * r + c;
            p2[0] = r;
            p2[1] = c;
            if (distance(p1, p2) <= radius)
                pixels[i] = 0;
            if (pixels[i] > newMaxIntens)
                newMaxIntens = pixels[i];
        }
    }

    sprintf(header[3], "%i", newMaxIntens);
    return (newMaxIntens == oldMaxIntens) ? 0 : 1;
}

/**
 *  Function Name:
 *      pgmDrawLine()
 *      pgmDrawLine() draws a straight line in the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image processing on GPU.
 *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 2D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      p1row specifies the row number of the start point of the line segment.
 *  @param[in]      p1col specifies the column number of the start point of the line segment.
 *  @param[in]      p2row specifies the row number of the end point of the line segment.
 *  @param[in]      p2col specifies the column number of the end point of the line segment.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawLine(int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col) {
    if (pixels == NULL || header == NULL)
        return 1;

    int oldMaxIntens = 0;
    int newMaxIntens = 0;
    int i = 0;
    sscanf(header[3], "%i", &oldMaxIntens);

    // avoid a divide by zero error by not calculating the slope
    if (p1col == p2col) {
        if (!intInRange(p1col, 0, numCols, UPPER_INCLUSIVE | LOWER_INCLUSIVE))
            return 0;

        int startRow = intMin(p1row, p2row);
        int endRow = intMax(p1row, p2row);
        int curRow = startRow;
        for (; curRow < endRow; ++curRow) {
            // make sure this pixel is actually within the image
            if (!intInRange(curRow, 0, numRows, UPPER_INCLUSIVE | LOWER_INCLUSIVE))
                continue;
            i = numRows * curRow + p1col;


            pixels[i] = 0;
            if (pixels[i] > newMaxIntens)
                newMaxIntens = pixels[i];
        }
    }
    else // we don't have a vertical line
    {
        int p1[2] = {0, 0};
        int p2[2] = {0, 0};

        if (p1col < p2col) {
            p1[0] = p1row;
            p1[1] = p1col;
            p2[0] = p2row;
            p2[1] = p2col;
        }
        else {
            p1[0] = p2row;
            p1[1] = p2col;
            p2[0] = p1row;
            p2[1] = p1col;
        }

        double slope = (p2[0] - p1[0]) / (p2[1] - p1[1]);
        if(DEBUG)
            printf("slope: %lf\n", slope);

        int thisCol = p1[1];
        for (; thisCol < numCols; ++thisCol) {

            int relativeCol = thisCol - p1[1];
            int thisRow = relativeCol * slope + p1[0];

            // make sure this pixel is actually within the image
            if (!intInRange(thisRow, 0, numRows - 1, UPPER_INCLUSIVE | LOWER_INCLUSIVE))
                continue;
            if (!intInRange(thisCol, 0, numCols - 1, UPPER_INCLUSIVE | LOWER_INCLUSIVE))
                continue;
            //if(DEBUG)
            //    printf("plot(%d, %d)\n", thisRow, thisCol);

            i = numRows * thisRow + thisCol;
            pixels[i] = 0;
            if (pixels[i] > newMaxIntens)
                newMaxIntens = pixels[i];

        }
    }

    sprintf(header[3], "%i", newMaxIntens);
    return (newMaxIntens == oldMaxIntens) ? 0 : 1;
}

double distance(int p1[], int p2[]) {
    if (p1 == NULL || p2 == NULL)
        return 0.0;

    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

/**
 *  Function Name;
 *      intInRange()
 *      intInRange() returns true if the target is within bound1 and bound2 (inclusivity can be checked if
 *                   LOWER_INCLUSIVE or UPPER_INCLUSIVE are included in the flags argument)
 */
int intInRange(int target, int bound1, int bound2, int flags) {
    int lowerBound = intMin(bound1, bound2);
    int upperBound = intMax(bound1, bound2);
    int result = ((target > lowerBound && target < upperBound) || (target == lowerBound && (flags & LOWER_INCLUSIVE)) ||
                  (target == upperBound && (flags & UPPER_INCLUSIVE)));
/*
    if (DEBUG) {
        char *intervalChars = "([)]";
        char *booleanStrings[] = {"false", "true"};
        char openIntervalChar = intervalChars[(flags & LOWER_INCLUSIVE) != 0];
        char closeIntervalChar = intervalChars[((flags & UPPER_INCLUSIVE) != 0) + 2];
        printf("%d in %c%d %d%c : %s\n", target, openIntervalChar, lowerBound, upperBound, closeIntervalChar,
               booleanStrings[result]);
    }
*/
    return result;
}
