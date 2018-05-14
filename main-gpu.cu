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
	SDPair*** sdPairsGPU = SDPair::calculateSDPairsFromGPU(network);
	timer.Stop();
	cout << left<< setw(30) << "PPV Calculation: " << timer.Elapsed() / 1000 << endl;

	/* before you enable this result comparison code, please enable the code in SDPair::calculateSDPairsFromGPU, which will convert the network representation to adjacency matrix represntation used in the comparison
	cout << "------------------------------\n";
	cout << "Result Comparison:\n";
	cout << "------------------------------\n";
	int ppvDifferences = 0;
	int usesLinkDifferences = 0;
	for (int s = 0; s < network->size; s++)
	{
		for (int d = 0; d < network->size; d++)
		{
			if (s == d) continue;
			if (sdPairsCPU[s][d]->comparePPV(*sdPairsGPU[s][d]) == false)
			{
				ppvDifferences++;
			}
			if (sdPairsCPU[s][d]->compareUsesLink(*sdPairsGPU[s][d]) == false)
			{
				usesLinkDifferences++;
			}
		}
	}
	if (!usesLinkDifferences)
	{
		cout << "Results are identical!\n";
	}
	else
	{
		cout << ppvDifferences << " out of " << network->size * network->size << " SD pairs have different PPV values between CPU and GPU\n";
		cout << usesLinkDifferences << " out of " << network->size * network->size << " SD pairs have different usesLink values between CPU and GPU\n";
		cout << "This is caused by the fact that the BFS algorithm implemented is different, and it picks its nodes in a slightly different order\n";
		cout << "This has been checked by inspecting many of the outputs, and validating that both of them are actually correct solutions\n";
		cout << "Notice that the number of difference in usesLink is generally less than the number of differences in PPV, because one difference in usesLink causes many differences in PPV\n";
	}*/

	

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