#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <stdlib.h>
using namespace std;

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cerr<< "Please provide the size of the network\n";
		return -1;
	}
	if (argc < 3)
	{
		cerr<< "Please provide the name of the file to which you want to write\n";
		return -1;
	}
	int minBandwidth = 1;
	int maxBandwidth = 20;
	if (argc >= 4)
	{
		minBandwidth = atoi(argv[3]);
	}
	if (argc >= 5)
	{
		maxBandwidth = atoi(argv[4]);
	}

	if (minBandwidth > maxBandwidth)
	{
		cerr<< "The minimum bandwidth "<< minBandwidth
			<< " is greater than the maximum bandwidth "<< maxBandwidth<< "!\n";
		return -1;
	}

	float sparsity = 0.7;
	if (argc >= 6)
	{
		sparsity = atof(argv[5]);
	}

	if (sparsity <= 0 || sparsity >= 1)
	{
		cerr<< "The sparsity "<< sparsity<< " is invalid. Please choose a sparsity between 0 and 1 exlusive\n";
	}

	int networkSize = atoi(argv[1]);
	
	ofstream fout;
	fout.open(argv[2]);
	if (!fout.is_open())
	{
		cerr<< "Failed to open file "<< argv[1]<< endl;
		return -1;
	}

	fout<< networkSize<< endl;

	srand(time(0));
	for (int i = 0; i < networkSize; i++)
	{
		for (int j = 0; j < networkSize; j++)
		{
			float hasConnectionProbability = (float)rand() / RAND_MAX;
			int bandwidth;
			if (i == j)
			{
				bandwidth = 0;	// We won't have self-links
			}
			else {
				if (hasConnectionProbability <= sparsity)
				{
					bandwidth = 0;
				}
				else
				{
					bandwidth = (rand() % (maxBandwidth + 1 - minBandwidth)) + minBandwidth;				
				}
			}
			fout<< bandwidth;
			if (j != networkSize - 1)
			{
				fout<< " ";
			}
		}
		fout<< endl;
	}

	fout.close();
}