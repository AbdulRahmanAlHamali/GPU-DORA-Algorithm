#ifndef SDPAIR_H
#define SDPAIR_H
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "Network.h"
#include "findShortestPath.h"
#include "uses-link.kernel.cuh"

class SDPair
{
private:
	Network* _network;
	bool** _usesLink;

	void _calculateUsesLink()
	{
		// Initialize _usesLink
		this->_usesLink = new bool*[this->_network->size];
		for (int i = 0; i < this->_network->size; i++)
		{
			this->_usesLink[i] = new bool[this->_network->size];
			for (int j = 0; j < this->_network->size; j++)
			{
				this->_usesLink[i][j] = false;
			}
		}

		int** copyOfNetwork = this->_network->getCopyOfNetwork();

		do
		{
			std::vector<int> previousVector
				= BFS(copyOfNetwork, this->_network->size, this->source, this->destination);

			if (previousVector[this->destination] == -1) break;

			int current = this->destination;
			while (current != this->source)
			{
				int prev = previousVector[current];
				this->_usesLink[prev][current] = true;
				copyOfNetwork[prev][current] = 0;
				current = prev;
			}
		} while (hasNoneZeroValues(copyOfNetwork[source], this->_network->size));

		for (int i = 0; i < this->_network->size; i++)
		{
			delete copyOfNetwork[i];
		}
		delete copyOfNetwork;
	}

	bool hasNoneZeroValues(int* row, int rowSize)
	{
		for (int i = 0; i < rowSize; i++)
		{
			if (row[i])
				return true;
		}
		return false;
	}

public:
	int source;
	int destination;
	float** ppv;
	static bool usesGpu;

	SDPair(int source, int destination, Network* network) :
		source(source), destination(destination), _network(network)
	{
		this->_usesLink = NULL;
		this->ppv = NULL;
#ifndef GPU
		this->_calculateUsesLink();
#endif
	}

	~SDPair()
	{
		if (this->_usesLink != NULL)
		{
			for (int i = 0; i < this->_network->size; i++)
			{
				delete this->_usesLink[i];
			}
			delete this->_usesLink;
		}
		if (this->ppv != NULL)
		{
			for (int i = 0; i < this->_network->size; i++)
			{
				delete this->ppv[i];
			}
			delete this->ppv;
		}
	}

	void calculatePPV(int** globalPPV)
	{
		int networkSize = this->_network->size;

		// Will be used later to normalize the PPV
		int min = 0;
		int max = 0;

		this->ppv = new float*[networkSize];
		for (int i = 0; i < networkSize; i++)
		{
			this->ppv[i] = new float[networkSize];
			for (int j = 0; j < networkSize; j++)
			{
				this->ppv[i][j] = globalPPV[i][j];
				if (this->_usesLink[i][j])
				{
					this->ppv[i][j] -= 2;
				}

				if (this->ppv[i][j] < min || min == 0)
				{
					min = this->ppv[i][j];
				}
				if (this->ppv[i][j] > max)
				{
					max = this->ppv[i][j];
				}
			}
		}

		for (int i = 0; i < networkSize; i++)
		{
			for (int j = 0; j < networkSize; j++)
			{
				ppv[i][j] = (ppv[i][j] - min) * (100.0 / (max - min));
			}
		}
	}

	static int** calculateGlobalPPV(SDPair*** allSDPairs, Network* network)
	{
		int networkSize = network->size;

		int** globalPPV = new int*[networkSize];
		for (int i = 0; i < networkSize; i++)
		{
			globalPPV[i] = new int[networkSize];
			for (int j = 0; j < networkSize; j++)
			{
				globalPPV[i][j] = 0;
				if (network->network[i][j] == 0)	continue;	// There is no link here
				for (int s = 0; s < networkSize; s++)
				{
					for (int d = 0; d < networkSize; d++)
					{
						if (allSDPairs[s][d]->_usesLink[i][j])
						{

							globalPPV[i][j] += 1;
						}
					}
				}
			}
		}

		return globalPPV;
	}

#ifdef GPU

	static int* d_vertexArray;
	static int* d_edgeArray;

	static void prepareGPU(Network* network) {
		// Restructure the network graph
		int networkSize = network->size;
		int* vertexArray = network->parallelRepresentation.vertices;
		int* edgeArray = network->parallelRepresentation.edges;
		int numberOfEdges = network->parallelRepresentation.numberOfEdges;

		cudaMalloc(&d_vertexArray, sizeof(int) * networkSize);
		cudaMalloc(&d_edgeArray, sizeof(int) * numberOfEdges);

		cudaMemcpy(d_vertexArray, vertexArray, sizeof(int) * networkSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_edgeArray, edgeArray, sizeof(int) * numberOfEdges, cudaMemcpyHostToDevice);
	}

	static void calculateUsesLinkGPU(Network* network)
	{
		dim3 blockSize(network->size, 1, 1);
		dim3 blocksPerGrid(network->size, network->size, 1);
		usesLinkKernel << <blocksPerGrid, blockSize, sizeof(int) * (4*network->size + network->parallelRepresentation.numberOfEdges) + sizeof(bool) >> > (SDPair::d_vertexArray, SDPair::d_edgeArray, network->size, network->parallelRepresentation.numberOfEdges);
	}
#endif
};

bool SDPair::usesGpu = false;
int* SDPair::d_vertexArray = NULL;
int* SDPair::d_edgeArray = NULL;

#endif