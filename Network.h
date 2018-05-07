#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include <string.h>

struct ParallelRepresenation
{
	int* vertices;
	int* edges;
	int numberOfEdges;
};

class Network
{
private:
	void _calculateParallelRepresentation()
	{
		std::vector<int> edgeVector;
		int* vertexArray = new int[this->size];

		for (int i = 0; i < this->size; i++)
		{
			vertexArray[i] = edgeVector.size();
			for (int j = 0; j < this->size; j++)
			{
				if (this->network[i][j] != 0)
				{
					edgeVector.push_back(j);
				}
			}
		}

		this->parallelRepresentation.vertices = vertexArray;

		this->parallelRepresentation.edges = new int[edgeVector.size()];
		memcpy(this->parallelRepresentation.edges, &edgeVector[0], sizeof(int) * edgeVector.size());

		this->parallelRepresentation.numberOfEdges = edgeVector.size();
	}
public:
	int** network;
	int size;
	struct ParallelRepresenation parallelRepresentation;

	Network(int** network, int size) :
		network(network), size(size)
	{
		this->_calculateParallelRepresentation();
	}

	int** getCopyOfNetwork()
	{
		int** copy = new int*[this->size];
		for (int i = 0; i < this->size; i++)
		{
			copy[i] = new int[this->size];
			for (int j = 0; j < this->size; j++)
			{
				copy[i][j] = this->network[i][j];
			}
		}

		return copy;
	}
};

#endif