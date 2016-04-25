
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] );


__global__ void gpuDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, int n);
__global__ void gpuDrawCircle(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, int n);
__global__ void gpuDrawLine(int *pixels, int noRows, int noCols, int startRow, int startCol, int vertical, float slope, int noSamples);
