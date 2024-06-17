## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Problem Description](#problem-description)
4. [EDA](#eda)
5. [Model Training](#model-training)
6. [Parameter Tuning](#parameter-tuning)
7. [Dependency and Environment Management](#dependency-and-environment-management)
8. [Containerization](#containerization)

# Introduction
Classification model of the energy consumption with the data collected from a smart small-scale steel industry in South Korea.<br> 
I took the data from UCI Machine Learning Repository.<br>

# Methodology

In this work, we used the dataset from the following research paper<br>
[@sathishkumar2023steel](https://doi.org/10.24432/C52G8C)<br><br>
The dataset is provided in the current repository. Here is the link [Steel_industry_data.csv](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/Steel_industry_data.csv)
<br>

```
wget 'https://raw.githubusercontent.com/Hokfu/Energy-Consumption-Model/main/Steel_industry_data.csv'
```

# Problem Description
A steel company has a few challenges apart from market competition like Increased energy Costs, downtime, inefficient resource allocation, maintenance, and regulatory compliance <br><br>
Problem: If the company does not know which conditions lead to high energy consumption and which ones lead to low and medium energy loads, those challenges will become serious problems. <br><br>
Opportunity: Vice versa, if the company can predict the energy consumption of a process in advance, it can improve in the challenges above, and can gain market advantage.
<br>
<br>
# EDA
Firstly, I tried to find the relation between numerical features and the target we want to know which is energy load type.
![relationship between numerical features and target variable](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/relationship%20between%20%20numerical%20features%20and%20target.png)
<br>
We can see clearly that NSM impacts  the most to the load type by checking relations in above violin plot. 
Violin plot or box plot can be used to find out the distribution of numerical features. In this case, I checked the distribution of each numerical features relating to each load type. 
<br>
<br>
<br>
![feature importance](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/finding%20feature%20importance.png) 
<br>
It is more obvious when we check feature importance while training the random forest model.
<br>
# Model Training
I trained with two models - logistic regression and random forest. Overall, random forest model seems to work better so I chose it as the final model.
<br>
<br>

# Parameter Tuning
Maximum depth and minimum sample leaves are tuned in a loop to find the best values.
<br>
![max depth tuning](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/finding%20best%20maximum%20depth.png)
![min sample leaves tuning](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/finding%20best%20min%20sample%20leaves.png)
<br>
<br>

# Dependency and Environment Management
For notebook and model training(train.py) <br>
Use conda or any environment. For conda environment, <br>

```
conda create -n 'environment-name' python=3.9.18
```

<br>
Activate conda environment<br>

```
conda activate 'environment-name'
```

<br>

```
pip install -r requirements.txt
```

to install requirements. 
<br>
<br>
For model prediction<br>
Use pipenv<br>

```
pipenv install numpy scikit-learn==1.3.0 gunicorn flask
```

# Containerization
<br>
For container building 
<br>

``` 
docker build -t <container_name> .
```
<br>
For container running
<br>

```
docker run -it --rm -p 9696:9696 <container_name>
```
<br>
Then, use another terminal and run predict_test.py to check the model.
<br>

# Deployment

In [Render](https://render.com/), create account, create a new web service, and deploy the container.










