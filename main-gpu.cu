#define GPU

#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <iomanip>
#include "SDPair.h"
#include "Network.h"
#include "GpuTimer.h"

using namespace std;


Network* readNetworkDescriptionFromFile(string fileName);

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cerr<< "Please provide name of file containing network description\n";
		return -1;
	}

	string fileName(argv[1]);
	Network* network;
	try 
	{
		network = readNetworkDescriptionFromFile(fileName);
	}
	catch(exception& e)
	{
		cerr<< e.what()<< endl;
		return -1;
	}

	// OFFLINE STAGE
	cout<< "Offline Stage CPU Performance:\n";
	cout<< "------------------------------\n";
	clock_t start = clock();
	
	SDPair*** sdPairsCPU;
	sdPairsCPU = new SDPair**[network->size];
	for (int i = 0; i < network->size; i++)
	{
		sdPairsCPU[i] = new SDPair*[network->size];
		for (int j = 0; j < network->size; j++)
		{
			sdPairsCPU[i][j] = new SDPair(i, j, network);
		}
	}
	
	int** globalPPVCPU = SDPair::calculateGlobalPPV(sdPairsCPU, network);
	for (int i = 0; i < network->size; i++)
	{
		for (int j = 0; j < network->size; j++)
		{
			if (i == j) continue;
			sdPairsCPU[i][j]->calculatePPV(globalPPVCPU);
		}
	}
	cout<< left<< setw(30)<<"PPV Calculation: "<< (clock() - start)/(double)CLOCKS_PER_SEC<< endl;

	// OFFLINE STAGE
	cout << "------------------------------\n";
	cout<< "Offline Stage GPU Performance:\n";
	cout<< "------------------------------\n";
	GpuTimer timer;
	timer.Start();
	SDPair::prepareGPU(network);
	int** globalPPVGPU;
	SDPair*** sdPairsGPU = SDPair::calculateSDPairsFromGPU(network, globalPPVGPU);
	for (int s = 0; s < network->size; s++)
	{
		for (int d = 0; d < network->size; d++)
		{
			if (s == d) continue;
			sdPairsGPU[s][d]->calculatePPV(globalPPVGPU);
		}
	}
	timer.Stop();
	cout << left<< setw(30) << "PPV Calculation: " << timer.Elapsed() / 1000 << endl;

	return 0;
}

Network* readNetworkDescriptionFromFile(string fileName)
{
	ifstream fin;
	fin.open(fileName.c_str());

	if (!fin.is_open())
	{
		throw runtime_error("Failed to open file " + fileName);
	}

	int size;
	fin>> size;
	int** result = new int*[size];
	for (int i = 0; i < size; i++)
	{
		result[i] = new int[size];
		for (int j = 0; j < size; j++)
		{
			fin>> result[i][j];
		}
	}

	fin.close();

	return new Network(result, size);
}