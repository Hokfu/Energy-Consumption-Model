# Energy-Consumption-Model
Classification model of the energy consumption with the data collected from a smart small-scale steel industry in South Korea.<br> 
I took the data from UCI Machine Learning Repository.<br>
Citation : [V E,Sathishkumar, Shin,Changsun, and Cho,Yongyun. (2023). Steel Industry Energy Consumption. UCI Machine Learning Repository. [https://doi.org/10.24432/C52G8C]. <br>
The dataset is provided in the current repository. Here is the link [Steel_industry_data.csv](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/Steel_industry_data.csv)
<br>
or wget link [https://raw.githubusercontent.com/Hokfu/Energy-Consumption-Model/main/Steel_industry_data.csv]
<br>
<br>
# Problem Description
A steel company has a few challenges apart from market competition like Increased energy Costs, downtime, inefficient resource allocation, maintenance, and regulatory compliance <br>
Problem: If the company does not know which conditions lead to high energy consumption and which ones lead to low and medium energy loads, those challenges will become serious problems. <br>
Opportunity: Vice versa, if the company can predict the energy consumption of a process in advance, it can improve in the challenges above, and can gain market advantage.
<br>
<br>
# EDA
Analysis of the relationship between feature variables and target variable
<br>
<br>
# Model Training
Logistic regression<br>
Random Forest
<br>
<br>
# Dependency and Environment Management
For notebook and model training(train.py) <br>
Please check [requirements.txt](https://github.com/Hokfu/Energy-Consumption-Model/blob/main/requirements.txt) for the required versions.
<br>
For model prediction<br>
Use pipenv<br>
$pipenv install numpy scikit-learn==1.3.0 gunicorn flask
<br>
No need to install this for testing containerization. For containerization, check below.
<br>
<br>
# Containerization
<br>
For container building 
<br>
$Docker build -t 'container_name' .<br>
e.g. $Docker build -t energy-consumption .
<br>
For container running
<br>
$Docker run -it --rm -p 9696:9696 'container_name'<br>
e.g. $Docker run -it --rm -p 9696:9696 energy-consumption<br>
Then, use another terminal and run predict_test.py to check the model.









