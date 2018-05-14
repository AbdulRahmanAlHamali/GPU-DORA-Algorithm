#include <cuda_runtime.h>

__global__ void usesLinkKernel(int* vertices, int* edges, bool* usesLink, int* ppv, int networkSize, int numberOfEdges)
{
	extern __shared__ char shared[];
	int* sharedVertices = (int*)shared;
	int* sharedEdges = (int*)shared + networkSize;
	int* previous = (int*)shared + (networkSize + numberOfEdges);
	int* level = (int*)shared + (2 * networkSize + numberOfEdges);
	int* current = (int*)shared + (3 * networkSize + numberOfEdges);
	bool* done = (bool*)((int*)shared + (3 * networkSize + numberOfEdges + 1));

	// Each block will work on one SD pair
	int source = blockIdx.y;
	int destination = blockIdx.x;

	if (source == destination)
	{
		return;
	}

	// Start by loading the network into shared memory
	for (int i = 0; i < numberOfEdges; i += 1024)
	{
		if (i + threadIdx.x < networkSize)
		{
			sharedVertices[i + threadIdx.x] = vertices[i + threadIdx.x];
		}
		if (i + threadIdx.x < numberOfEdges)
		{
			sharedEdges[i + threadIdx.x] = edges[i + threadIdx.x];
		}
	}
	
	do
	{
		__syncthreads();
		if (threadIdx.x < networkSize)
		{
			if (threadIdx.x != source)
			{
				level[threadIdx.x] = -1;
				previous[threadIdx.x] = -1;
			}
			else
			{
				*done = true;
				*current = 0;
				level[source] = 0;
			}
		}

		do
		{	
			__syncthreads();
			if (threadIdx.x == 0)
			{
				*done = true;
			}
			__syncthreads();
			if (threadIdx.x < networkSize && level[threadIdx.x] == *current)
			{
				int start = sharedVertices[threadIdx.x];
				int end = threadIdx.x < networkSize - 1? sharedVertices[threadIdx.x + 1] - 1 : numberOfEdges - 1;
						
				for (int i = start; i <= end; i++) 
				{
					if (sharedEdges[i] == -1) continue;
					int target = sharedEdges[i];

					if (level[target] == -1)
					{
						atomicCAS(&previous[target], -1, threadIdx.x);
						level[target] = *(current) + 1;
						*done = false;
					}

				}
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				if (previous[destination] != -1)
					*done = true;
				(*current)++;
			}
			__syncthreads();
		} while (!(*done));
		
		if (previous[destination] == -1) break;
		
		if (threadIdx.x == 0)
		{
			int curr = destination;
			while (curr != source)
			{
				int prev = previous[curr];
				int start = sharedVertices[prev];
				int end = prev < networkSize - 1? sharedVertices[prev + 1] - 1 : numberOfEdges - 1;
				for (int i = start; i <= end; i++) 
				{
					if (sharedEdges[i] == curr)
					{
						usesLink[source * networkSize * numberOfEdges + destination * numberOfEdges  + i] = true;
						sharedEdges[i] = -1;
						break;
					}

				}
				curr = prev;
			}
		}

	} while(true);

	__syncthreads();

	for (int i = 0; i < numberOfEdges; i += 1024)
	{
		if (i + threadIdx.x < numberOfEdges && usesLink[source * networkSize * numberOfEdges + destination * numberOfEdges + i + threadIdx.x])
		{
			atomicAdd(&ppv[i + threadIdx.x], 1);
		}
	}
	

}