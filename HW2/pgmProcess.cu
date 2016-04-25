//#include "pgmUtility.h"
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] )
{
    int distanceX = p2[1] - p1[1];
    int distanceY = p2[0] - p1[0];
    //return sqrt((float)(distanceX * distanceX) + (float)(distanceY * distanceY));
    return 0.0;
}

int  pgmDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, char **header) {
    return 0;
}

int pgmDrawCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header) {
    return 0;
}

int pgmDrawLine(int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col){
    return 0;
}



