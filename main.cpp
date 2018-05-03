#include <iostream>
#include <string>
#include <fstream>
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