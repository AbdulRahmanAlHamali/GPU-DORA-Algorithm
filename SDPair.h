#ifndef SDPAIR_H
#define SDPAIR_H
#include "Network.h"
#include "findShortestPath.h"

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

	}
public:
	int source;
	int destination;

	SDPair(int source, int destination, Network* network):
		source(source), destination(destination), _network(network)
	{
		this->_calculateUsesLink();
	}
};

#endif