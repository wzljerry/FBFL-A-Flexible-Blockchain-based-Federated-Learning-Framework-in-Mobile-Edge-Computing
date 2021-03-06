# FBFL-A-Flexible-Blockchain-based-Federated-Learning-Framework-in-Mobile-Edge-Computing
This framework is based on our previous research. We built a federated learning system and blockchain system from scratch. It is a simulation platform for researchers and companies to apply blockchain in a federated learning framework, and it can be applied in MEC. It is a basic model that provides most of the functions, such as attacks, data distribution analysis, blockchain consensus, etc. We will update it in the future.

Since this project is based on our previous theoretical research and some parts of it are currently completed but not officially published, we cannot publish the complete code.

## How to run it?
First, we will clarify the dependencies of this project.
1. python==3.9
2. keras==2.8.0
3. tensorflow==2.8.0
4. numpy==1.21.6
5. pandas==1.3.5
6. matplotlib==3.2.2
7. flask==1.1.4
8. requests==2.26.0

We recommend you to use virtual environment on Linux.

1. install all the required packages:

pip install -r requirement.txt

2. run the federated learning system in terminal:

python3 blfec.py

3. run the blockchain system in terminal:

python3 blockchain.py

Then in another terminal, run

curl  http://127.0.0.1:5000/chain

## What can it do?

1. Test different data on federated learning.

We can test different datasets with IID or Non-IID data.

2. Test different aggregation methods of federated learning.

We provdide some popular aggregation methods.

3. Test different blockchain protocols.

5. Test different attacks.

## What we will do in the future?
1. We will implement it on numerous devices.
2. We will update the platform, providing more functions such as defense strategies.

