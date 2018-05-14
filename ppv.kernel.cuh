#include <cuda_runtime.h>

__global__ void ppvKernel(bool* usesLink, int* globalPPV, int* ppv, int networkSize, int numberOfEdges)
{
	int source = blockIdx.y;
	int destination = blockIdx.x;

	for (int i = 0; i < numberOfEdges; i += 1024)
	{
		if (i + threadIdx.x < numberOfEdges)
		{
			if (usesLink[source * networkSize * numberOfEdges + destination * numberOfEdges + i + threadIdx.x])
			{
				ppv[source * networkSize * numberOfEdges + destination * numberOfEdges + i + threadIdx.x] = globalPPV[i + threadIdx.x] + 2;
			}
			else
			{
				ppv[source * networkSize * numberOfEdges + destination * numberOfEdges + i + threadIdx.x] = globalPPV[i + threadIdx.x];
			}
		}
	}

}