#include <cuda_runtime.h>

__global__ void usesLinkKernel(int* vertices, int* edges, bool* usesLink, int* ppv, int networkSize, int numberOfEdges)
{
	extern __shared__ char shared[];
	int* sharedVertices = (int*)shared;
	int* sharedEdges = (int*)shared + networkSize;
	int* previous = (int*)shared + (networkSize + numberOfEdges);
	bool* visited = (bool*)((int*)shared + (2*networkSize + numberOfEdges));
	bool* frontier = (bool*)((int*)shared + (2 * networkSize + numberOfEdges)) + networkSize;
	bool* nextFrontier = (bool*)((int*)shared + (2 * networkSize + numberOfEdges)) + 2 * networkSize;
	
	// Each block will work on one SD pair
	int source = blockIdx.y;
	int destination = blockIdx.x;

	if (source == destination)
	{
		return;
	}

	// Start by loading the network into shared memory
	if (threadIdx.x < networkSize)
	{
		sharedVertices[threadIdx.x] = vertices[threadIdx.x];
	}
	if (threadIdx.x < numberOfEdges)
	{
		sharedEdges[threadIdx.x] = edges[threadIdx.x];
	}

	do
	{
		if (threadIdx.x < networkSize)
		{
			if (threadIdx.x != source)
			{
				frontier[threadIdx.x] = false;
				nextFrontier[threadIdx.x] = false;
				previous[threadIdx.x] = -1;
			}
			else
			{
				frontier[source] = true;
			}
			visited[threadIdx.x] = false;
		}
		
		__syncthreads();
		if (threadIdx.x == 0)
		{
			bool done = false;
			while (!done)
			{
				done = true;
				for (int t = 0; t < networkSize; t++)
				{
					if (frontier[t] == true && visited[t] == false)
					{
						frontier[t] = false;
						visited[t] = true;
						
						int start = sharedVertices[t];
						int end = t < networkSize - 1? sharedVertices[t + 1] - 1 : numberOfEdges - 1;
						
						for (int i = start; i <= end; i++) 
						{
							if (sharedEdges[i] == -1) continue;
							int target = sharedEdges[i];

							if (visited[target] == false && frontier[target] == false && nextFrontier[target] == false)
							{
								previous[target] = t;
								nextFrontier[target] = true;
								done = false;
							}

						}
					}
				}

				for (int i = 0; i < networkSize; i++)
				{
					frontier[i] = nextFrontier[i];
					nextFrontier[i] = false;
				}
				
				if (visited[destination] == true)
					done = true;
			}
		}
		__syncthreads();
		
		if (previous[destination] == -1) break;
		
		if (threadIdx.x == 0)
		{
			int current = destination;
			while (current != source)
			{
				int prev = previous[current];
				int start = sharedVertices[prev];
				int end = prev < networkSize - 1? sharedVertices[prev + 1] - 1 : numberOfEdges - 1;
				for (int i = start; i <= end; i++) 
				{
					if (sharedEdges[i] == current)
					{
						usesLink[source * networkSize * numberOfEdges + destination * numberOfEdges  + i] = true;
						sharedEdges[i] = -1;
						break;
					}

				}
				current = prev;
			}
		}

	} while(true);

	__syncthreads();

	if (threadIdx.x < numberOfEdges)
	{
		atomicAdd(&ppv[threadIdx.x], usesLink[source * networkSize * numberOfEdges + destination * numberOfEdges + threadIdx.x]);
	}

}