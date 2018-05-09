#ifndef SDPAIR_H
#define SDPAIR_H
#include <iostream>
#include <vector>
#ifdef GPU
#include <cuda_runtime.h>
#endif
#include <ctime>
#include "Network.h"
#include "findShortestPath.h"
#ifdef GPU
#include "uses-link.kernel.cuh"
#endif

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
		} while (_hasNoneZeroValues(copyOfNetwork[source], this->_network->size));		

		for (int i = 0; i < this->_network->size; i++)
		{
			delete[] copyOfNetwork[i];
		}
		delete[] copyOfNetwork;
	}

	bool _hasNoneZeroValues(int* row, int rowSize)
	{
		for (int i = 0; i < rowSize; i++)
		{
			if (row[i])
				return true;
		}
		return false;
	}
#ifdef GPU
	void _convertUsesLinkFromGpuToStandardFormat(bool* gpuUsesLink)
	{
		int networkSize = this->_network->size;
		this->_usesLink = new bool*[networkSize];
		
		int gpuIndex = 0;
		for (int s = 0; s < networkSize; s++)
		{
			this->_usesLink[s] = new bool[networkSize];
			for (int d = 0; d < networkSize; d++)
			{
				this->_usesLink[s][d] = this->_network->network[s][d] != 0;
				if (this->_usesLink[s][d])
				{
					this->_usesLink[s][d] = gpuUsesLink[gpuIndex];
					gpuIndex++;
				}
			}
		}
	}
#endif
public:
	int source;
	int destination;
	float** ppv;

	SDPair(int source, int destination, Network* network) :
		source(source), destination(destination), _network(network)
	{
		this->_usesLink = NULL;
		this->ppv = NULL;
		this->_calculateUsesLink();
	}

#ifdef GPU
	SDPair(int source, int destination, Network* network, bool** usesLink):
		source(source), destination(destination), _network(network), _usesLink(usesLink)
	{
		this->ppv = NULL;
	}
#endif

	~SDPair()
	{
		if (this->_usesLink != NULL)
		{
			for (int i = 0; i < this->_network->size; i++)
			{
				delete[] this->_usesLink[i];
			}
			delete[] this->_usesLink;
		}
		if (this->ppv != NULL)
		{
			for (int i = 0; i < this->_network->size; i++)
			{
				delete[] this->ppv[i];
			}
			delete[] this->ppv;
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

		cudaMalloc(&SDPair::d_vertexArray, sizeof(int) * networkSize);
		cudaMalloc(&SDPair::d_edgeArray, sizeof(int) * numberOfEdges);
		
		cudaMemcpy(SDPair::d_vertexArray, vertexArray, sizeof(int) * networkSize, cudaMemcpyHostToDevice);
		cudaMemcpy(SDPair::d_edgeArray, edgeArray, sizeof(int) * numberOfEdges, cudaMemcpyHostToDevice);
		
	}

	static SDPair*** calculateSDPairsFromGPU(Network* network, int**&globalPPV)
	{
		int networkSize = network->size;
		int numberOfEdges = network->parallelRepresentation.numberOfEdges;

		bool* d_usesLink;
		cudaMalloc(&d_usesLink, sizeof(bool) * networkSize * networkSize * numberOfEdges);
		cudaMemset(d_usesLink, false, sizeof(bool) * networkSize * networkSize * numberOfEdges);

		int* d_globalPPV;
		cudaMalloc(&d_globalPPV, sizeof(int) * numberOfEdges);
		cudaMemset(d_globalPPV, 0, sizeof(int) * numberOfEdges);

		dim3 blockSize(networkSize > numberOfEdges? networkSize : numberOfEdges, 1, 1);
		dim3 blocksPerGrid(networkSize, networkSize, 1);
		usesLinkKernel << <blocksPerGrid, blockSize, sizeof(int) * (2* networkSize + numberOfEdges) + 3 * sizeof(bool) * networkSize >> >
			(SDPair::d_vertexArray, SDPair::d_edgeArray, d_usesLink, d_globalPPV, networkSize, numberOfEdges);

		bool* usesLink = new bool[networkSize * networkSize * numberOfEdges];
		cudaMemcpy(usesLink, d_usesLink, sizeof(bool) * networkSize * networkSize * numberOfEdges, cudaMemcpyDeviceToHost);

		int* gpuGlobalPPV = new int[numberOfEdges];
		cudaMemcpy(gpuGlobalPPV, d_globalPPV, sizeof(int) * numberOfEdges, cudaMemcpyDeviceToHost);
		
		SDPair*** result;
		SDPair*** result = new SDPair**[networkSize];
		globalPPV = new int*[networkSize];
		int gpuPPVIndex = 0;
		for (int s = 0; s < networkSize; s++)
		{
			globalPPV[s] = new int[networkSize];
			result[s] = new SDPair*[networkSize];
			for (int d = 0; d < networkSize; d++)
			{
				globalPPV[s][d] = 0;
				if (network->network[s][d] != 0)
				{
					globalPPV[s][d] = gpuGlobalPPV[gpuPPVIndex];
					gpuPPVIndex++;
				}

				bool** sdUsesLink = new bool*[networkSize];
				int gpuIndex = 0;
				for (int s2 = 0; s2 < networkSize; s2++)
				{
					sdUsesLink[s2] = new bool[networkSize];
					for (int d2 = 0; d2 < networkSize; d2++)
					{
						sdUsesLink[s2][d2] = false;
						if (network->network[s2][d2] != 0)
						{
							sdUsesLink[s2][d2] = usesLink[s * networkSize * numberOfEdges + d * numberOfEdges + gpuIndex];
							gpuIndex++;
						}
					}
				}
				result[s][d] = new SDPair(s, d, network, sdUsesLink);
			}
		}

		return result;

	}
#endif

	bool** getUsesLink()
	{
		return this->_usesLink;
	}
};

#ifdef GPU
int* SDPair::d_vertexArray = NULL;
int* SDPair::d_edgeArray = NULL;
#endif
#endif