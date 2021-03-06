# NAS-Implements

## 1.	Overview
NAS automates the manual process of designing neural network’s various tasks like classification and regression. 
In NAS for given datasets, 
•	Define a search space of architectures 
•	Use a search strategy like reinforcement learning, to find the best architecture for the dataset
•	Retrain the model (best architecture) on the dataset and find the performance. 
Below is the basic NAS workflow:

![image](https://user-images.githubusercontent.com/59950610/116515171-907eaf80-a8e9-11eb-95c4-7d77bda9820d.png)

#### Below is the NAS implemented methods & toolkits:
*	NNI (Neural Network Intelligence) 
  *	ENAS
  *	DARTS
  *	PDARTS
*	DeepHyper
*	Auto-Keras


##	How to run the codes
## 1.	ENAS:
  ### search architecture:
      python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'search' --last_activation 'softmax'

  ### retrain the best architecture:
      python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'retrain' --arch_path './final_architecture.json'

## 2.	DARTS:
  ### search architecture:
     python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'search' --last_activation 'softmax'

  ### retrain the best architecture:
     python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'retrain' --arch_path './final_architecture.json'

### 3.	PDARTS:
  ### search architecture: 
     python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'search' --last_activation 'softmax'

  ### retrain the best architecture
     python3 search.py --num_epochs 5 --input_shape 34 --out_shape 64 --train_mode 'retrain' --arch_path './final_architecture.json'

