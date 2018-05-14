# GPU-DORA-Algoritm
What is this project?
========
GPU Implementation of the offline stage of DORA algorithm for Software-Defined Networks ([see paper](https://github.com/AbdulRahmanAlHamali/GPU-DORA-Algorithm/blob/master/GPU%20Implementation%20of%20the%20Offline%20Stage%20of%20DORA%20Algorithm%20for%20SDN.pdf))  


The project was implemented as a course project for EECE696: Applied Parallel Programming in the American University of Beirut


Test case generation
========

A few test cases are already provided in test-networks directory. However if you wish to create your own test cases:

First, compile generate_sample_network.cpp, by using:
```
g++ generate_sample_network.cpp -o generate_sample_network
```
Then, we can call generate_sample_network as follows:
```
generate_sample_network network_size output_file [min_bandwidth] [max_bandwidth] [sparsity]
```
Where:
- network_size is the number of nodes in the network
- output_file is the file to write the network to
- min_bandwidth is optional (default 1), and represents the minimum bandwidth of the links in the network
- max_bandwidth is optional (default 20), and represents the maximum bandwidth of the links in the network
- sparsity is optional (default 0.7), and represents how sparce the network is (1 means no links, 0 means there are links from every node to every node)

Running the code
========

First, compile main-gpu.cu, by using:
```
nvcc main-gpu.cu -o main-gpu
```
Then run:
```
main network_file
```
where network_file is the file containing the network description
