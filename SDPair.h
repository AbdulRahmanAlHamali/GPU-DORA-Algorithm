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
#include "ppv.kernel.cuh"
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
public:
	int source;
	int destination;
	int** ppv;

	SDPair(int source, int destination, Network* network) :
		source(source), destination(destination), _network(network)
	{
		this->_usesLink = NULL;
		this->ppv = NULL;
		this->_calculateUsesLink();
	}

#ifdef GPU
	SDPair(int source, int destination, Network* network, bool** usesLink, int** ppv):
		source(source), destination(destination), _network(network), _usesLink(usesLink), ppv(ppv)
	{
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

		this->ppv = new int*[networkSize];
		for (int i = 0; i < networkSize; i++)
		{
			this->ppv[i] = new int[networkSize];
			for (int j = 0; j < networkSize; j++)
			{
				this->ppv[i][j] = globalPPV[i][j];
				if (this->_usesLink[i][j])
				{
					this->ppv[i][j] -= 2;
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

		auto error = cudaMalloc(&SDPair::d_vertexArray, sizeof(int) * networkSize);
		if (error != cudaSuccess)
		{
			std::cerr << "Error allocating vertex array: "<< cudaGetErrorString(error)<< std::endl;
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc(&SDPair::d_edgeArray, sizeof(int) * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error allocating edge array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(SDPair::d_vertexArray, vertexArray, sizeof(int) * networkSize, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cerr << "Error copying vertex array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(SDPair::d_edgeArray, edgeArray, sizeof(int) * numberOfEdges, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			std::cerr << "Error copying edge array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	static SDPair*** calculateSDPairsFromGPU(Network* network)
	{
		int networkSize = network->size;
		int numberOfEdges = network->parallelRepresentation.numberOfEdges;

		bool* d_usesLink;
		auto error = cudaMalloc(&d_usesLink, sizeof(bool) * networkSize * networkSize * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error allocating usesLink array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}
		error = cudaMemset(d_usesLink, false, sizeof(bool) * networkSize * networkSize * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error initializing usesLink array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		int* d_globalPPV;
		error = cudaMalloc(&d_globalPPV, sizeof(int) * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error allocating global PPV array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}
		error = cudaMemset(d_globalPPV, 0, sizeof(int) * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error initializing global PPV array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		int* d_PPV;
		error = cudaMalloc(&d_PPV, sizeof(int) * networkSize * networkSize * numberOfEdges);
		if (error != cudaSuccess)
		{
			std::cerr << "Error allocating PPV array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		error = cudaSuccess;
		int numberOfThreads = networkSize > numberOfEdges ? networkSize : numberOfEdges;
		numberOfThreads = numberOfThreads <= 1024 ? numberOfThreads : 1024;
		dim3 blockSize(numberOfThreads, 1, 1);
		dim3 blocksPerGrid(networkSize, networkSize, 1);
		
		usesLinkKernel << <blocksPerGrid, blockSize, sizeof(int) * (3* networkSize + numberOfEdges + 1) + 1 * sizeof(bool)>> >
			(SDPair::d_vertexArray, SDPair::d_edgeArray, d_usesLink, d_globalPPV, networkSize, numberOfEdges);
		
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			std::cerr << "Error executing usesLink kernel: " << cudaGetErrorString(error) << std::endl;
			std::cerr << "The following information might be helpful\n";
			std::cerr << "Number of blocks: " << networkSize * networkSize << std::endl;
			std::cerr << "Number of threads per block: " << numberOfThreads << std::endl;
			std::cerr << "Size of shared memory usage: " << sizeof(int) * (3 * networkSize + numberOfEdges + 1) + 1 * sizeof(bool) << std::endl;
			exit(EXIT_FAILURE);
		}

		cudaDeviceSynchronize();

		ppvKernel << <blocksPerGrid, blockSize>> >(d_usesLink, d_globalPPV, d_PPV, networkSize, numberOfEdges);

		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			std::cerr << "Error executing usesLink kernel: " << cudaGetErrorString(error) << std::endl;
			std::cerr << "The following information might be helpful\n";
			std::cerr << "Number of blocks: " << networkSize * networkSize << std::endl;
			std::cerr << "Number of threads per block: " << numberOfThreads << std::endl;
			std::cerr << "Size of shared memory usage: " << sizeof(int) * (3 * networkSize + numberOfEdges + 1) + 1 * sizeof(bool) << std::endl;
			exit(EXIT_FAILURE);
		}

		bool* usesLink = new bool[networkSize * networkSize * numberOfEdges];
		error = cudaMemcpy(usesLink, d_usesLink, sizeof(bool) * networkSize * networkSize * numberOfEdges, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			std::cerr << "Error copying usesLink array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		int* gpuPPV = new int[networkSize * networkSize * numberOfEdges];
		error = cudaMemcpy(gpuPPV, d_PPV, sizeof(int) * networkSize * networkSize * numberOfEdges, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			std::cerr << "Error copying globaPPV array: " << cudaGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		SDPair*** result = new SDPair**[networkSize];
		for (int s = 0; s < networkSize; s++)
		{
			result[s] = new SDPair*[networkSize];
			for (int d = 0; d < networkSize; d++)
			{
				bool** sdUsesLink = new bool*[networkSize];
				int** sdPPV = new int*[networkSize];
				int gpuIndex = 0;
				for (int s2 = 0; s2 < networkSize; s2++)
				{
					sdUsesLink[s2] = new bool[networkSize];
					sdPPV[s2] = new int[networkSize];
					for (int d2 = 0; d2 < networkSize; d2++)
					{
						sdUsesLink[s2][d2] = false;
						sdPPV[s2][d2] = 0;
						if (network->network[s2][d2] != 0)
						{
							sdUsesLink[s2][d2] = usesLink[s * networkSize * numberOfEdges + d * numberOfEdges + gpuIndex];
							sdPPV[s2][d2] = gpuPPV[s * networkSize * numberOfEdges + d * numberOfEdges + gpuIndex];
							gpuIndex++;
						}
					}
				}
				result[s][d] = new SDPair(s, d, network, sdUsesLink, sdPPV);
			}
		}

		return result;

	}
#endif

	bool comparePPV(SDPair& other)
	{
		for (int i = 0; i < this->_network->size; i++)
		{
			for (int j = 0; j < this->_network->size; j++)
			{
				if (this->ppv[i][j] != other.ppv[i][j])
				{
					return false;
				}
			}
		}
		return true;
	}

	bool compareUsesLink(SDPair& other)
	{
		for (int i = 0; i < this->_network->size; i++)
		{
			for (int j = 0; j < this->_network->size; j++)
			{
				if (this->_usesLink[i][j] != other._usesLink[i][j])
				{
					return false;					
				}
			}
		}
		return true;
	}

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