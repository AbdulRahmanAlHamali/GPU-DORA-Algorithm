#include <cuda_runtime.h>

__global__ void usesLinkKernel(int* vertices, int* edges, int networkSize, int numberOfEdges)
{
	extern __shared__ int shared[];
	int* sharedVertices = shared;
	int* sharedEdges = shared + sizeof(int)*networkSize;
	int* frontier = shared + sizeof(int)*(networkSize + numberOfEdges);
	int* visited = shared + sizeof(int)*(2 * networkSize + numberOfEdges);
	int* previous = shared + sizeof(int)*(3 * networkSize + numberOfEdges);
	bool* usesLink = shared + sizeof(int)*(4 * networkSize + numberOfEdges);
	bool* done = shared + sizeof(int)*(4 * networkSize +  numberOfEdges) + sizeof(bool) * numberOfEdges;

	// Each block will work on one SD pair
	int source = blockIdx.x;
	int destination = blockIdx.y;

	// Start by loading the network into shared memory
	if (threadIdx.x < networkSize)
	{
		sharedVertices[threadIdx.x] = vertices[threadIdx.x];
	}
	if (threadIdx.x < numberOfEdges)
	{
		sharedEdges[threadIdx.x] = edges[threadIdx.x];
		_usesLink[threadIdx.x] = false;
	}

	__syncthreads();
	
	// Now, we do BFS
	do
	{
		if (threadIdx.x < networkSize)
		{
			if (threadIdx.x != source)
			{
				frontier[threadIdx.x] = false;
				visited[threadIdx.x] = false;
				previous[threadIdx.x] = -1;
			}
			else
			{
				frontier[source] = true;
				*done = false;
			}
			visited[threadIdx.x] = false;
		}

		__syncthreads();
		if (threadIdx.x == 0)
		{
			while (!(*done))
			{
				*done = true;
				for (int t = 0; t < networkSize; t++)
				{
					if (frontier[t] == true && visited[t] == false)
					{
						frontier[t] = false;
						visited[t] = true;
						
						int start = sharedVertices[t];
						int end = t < networkSize - 1? (sharedVertices[t + 1] - 1 : numberOfEdges - 1);
						for (int i = start; i < end; i++) 
						{
							int target = sharedEdges[i];

							if (visited[target] == false)
							{
								prev[target] = t;
								frontier[target] = true;
								*done = false;
							}

						}
					}
				}
				
				if (visited[destination] == true)
					*done = true;
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
				int end = prev < networkSize - 1? (sharedVertices[prev + 1] - 1 : numberOfEdges - 1);
				for (int i = start; i < end; i++) 
				{
					if (sharedEdges[i] == current)
					{
						usesLink[i] = true;
						sharedEdges[i] = -1;
						break;
					}

				}
				current = prev;
			}
		}

	} while(true);

	

}