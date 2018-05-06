#ifndef FIND_SHORTEST_PATH
#define FIND_SHORTEST_PATH
#include <queue>
#include <vector>

/**
 * Runs BFS algorithm to find the shortest unweighted path from node s to node d
**/
std::vector<int> BFS(int** network, int networkSize, int source, int destination)
{
    std::vector<bool> visited(networkSize, false);
    std::vector<int> previous(networkSize, -1);
    std::queue<int> Q;
 	Q.push(source);
	visited[source] = true;
	
	while(!Q.empty())
	{
		int top = Q.front(); 
		Q.pop();

		for (int i = 0; i < networkSize; i++)
		{
			if (network[top][i] != 0 && !visited[i])
			{
				previous[i] = top;
				if (i == destination)
					return previous;
				Q.push(i);
				visited[i] = true;				
			}
		}
	}

	return previous;
}

// void findShortestPath(int s, int d, int** network, int networkSize)
// {
// 	// The distance to each node
// 	double* distance = new int[networkSize];
// 	// The previous node that leads to this node
// 	int* prevNode = new int[networkSize];
// 	// Whether each node is done or not
// 	bool* isDone = new bool[networkSize];

// 	for (int i = 0; i < networkSize; i++)
// 	{
// 		distance[i] = -1;
// 		prevNode[i] = -1;
// 		isDone[i] = false;
// 	}

// 	distance[s] = 0;

// 	int nodesDone = 0;

// 	while (nodesDone < networkSize)
// 	{
// 		int min = 0
// 		for (int i = 1; i < networkSize; i++)
// 		{
// 			if (!isDone[i] && distance[i] != -1 && distance[i] < distance[min])
// 			{
// 				min = i;
// 			}
// 		}

// 		isDone[min] = true;

// 		for (int i = 0; i < networkSize; i++)
// 		{
// 			if (i != min && !isDone[i])
// 			{
// 				double dist = distance[min] + (1.0 / )
// 			}
// 		}
// 	}

// }

#endif