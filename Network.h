#ifndef NETWORK_H
#define NETWORK_H

class Network
{
public:
	int** network;
	int size;

	Network(int** network, int size):
		network(network), size(size)
	{

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