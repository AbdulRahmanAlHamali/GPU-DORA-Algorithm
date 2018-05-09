#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <iomanip>
#include "SDPair.h"
#include "Network.h"
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
	cout<< "Offline Stage Performance:\n";
	cout<< "------------------------------\n";
	clock_t start = clock();

	SDPair*** sdPairs;
	sdPairs = new SDPair**[network->size];
	for (int i = 0; i < network->size; i++)
	{
		sdPairs[i] = new SDPair*[network->size];
		for (int j = 0; j < network->size; j++)
		{
			clock_t start2;
			sdPairs[i][j] = new SDPair(i, j, network);
		}
	}

	cout<< left<< setw(30)<<"Link Usage Calculation: "<< (clock() - start)/(double)CLOCKS_PER_SEC<< endl;
	start = clock();
	int** globalPPV = SDPair::calculateGlobalPPV(sdPairs, network);
	for (int i = 0; i < network->size; i++)
	{
		for (int j = 0; j < network->size; j++)
		{
			if (i == j) continue;
			sdPairs[i][j]->calculatePPV(globalPPV);
		}
	}
	cout<< setw(30)<<"PPV Calculation: "<< (clock() - start)/(double)CLOCKS_PER_SEC<< endl;

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