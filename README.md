# AI_Project
Managing a financial portfolio involves striking a balance between reducing risk and optimizing returns. Given the dynamic character of financial markets—their constantly fluctuating volatility, changing economic situations, and changing investor preferences—this is no simple task. Although portfolio management has been based on traditional procedures, these approaches frequently fail to meet the complexity of contemporary financial settings. With the "AI-Powered Portfolio Optimization" project, we hope to address these issues by applying state-of-the-art Probabilistic methods.

Due to high dimensionality, dynamic correlations among assets, and the need to trade off risk and return, portfolio optimization is more than straightforward calculations. The model needs to be flexible so that it can adapt to managing multiple assets, given that the relationship changes over time. With data being time-series data it has non-linear trends and patterns that demand AI modeling techniques. These complexities make the problem non-trival in nature.

Conventional approaches to portfolio optimization include the Capital Asset Pricing Model (CAPM), which assesses the connection between systematic risk and expected return, and the Markowitz Model, which concentrates on efficient frontier optimization. Other approaches have proven successful in particular situations, such as the GARCH Model for capturing time-varying volatility and the Black-Scholes Model for options pricing. They frequently fall short, nevertheless, in addressing the intricacies of dynamic markets.

In the proposed model we follow a framework leveraging AI and probablistic techniques to improve portfolio analysis and optimization. The approach focuses on defining a state space, asset prices, returns, volatilitty and market sentiments. The action space includes adjusting  asset weights, rebalancing strategies and tactically buying, selling or holding assets. Observations such as real time price data, trade volume and sentiment data make the model better. By using Markov chains, Monte Carlo simulations and Particle filters the model tries to transition between market states, assess different possible outcomes under uncertainty and optimize the portfolio based on it.

Our AI-driven methodology begins with data collection and preprocessing, where historical price data for stocks and bonds is gathered. We assign market states (Bull, Bear, Neutral) using Markov Chains based on portfolio returns. Monte Carlo simulations project portfolio performance under varying market conditions, while state transition analysis computes steady-state distributions of market conditions. Particle Filters enable real-time updates to dynamically optimize portfolio weights. Finally, we compare cumulative returns of real vs. optimized portfolios and analyze correlation matrices to ensure diversification.

To make this model I have followed the following steps - 
Step 1- Get historic data of the assets in the code we have stocks and bonds as our assets.
Step 2- Do basic Preprocessing of data like handling null or na values
Step 3- Then we assign the Market states (Bull, Bear or Neutral) to portfolio returns returrning the state vector and transtion matrix.
Step 4- Now we perform Monte Carlo simulation using the transtional probabilties, these simulations help us with performance under changing market conditions.
Step 5- Particle filter helps with optimizing portfolio weights. 
Step 6- Finally we compare cummulative returns of real vs optimized portfolios and analyze correlation between the selected assets in our optimized portfolio.

To add market sentiments into the model I had used the scraped market news data of last 2 months of all the assets and assigned market states to the news. After that I had calculated the transtion matrix, steady state, and then used thiese probabilities to do step 4, 5 and 6.

Now lets go over the code and the discuss the result from the two models above - 

Some of the denpendencies you will need for the code to run - 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
import seaborn as sns 
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

Now that dependencies are done lets get the data. 
Step 1- Historic Data
to get the data I have used yahoo finance,  by using thhe following code. 

assets_data = {}
for asset in assets:
  ticker = yf.Ticker(asset)
  assets_data[asset] = ticker.history(start=start_date, end=end_date)['Close']
data = pd.DataFrame(assets_data)

The code above is pretty self explainatory, I had declared a list of assets which had around 44 stocks and 2 bonds all are listed on New York Stock Exchange, if someone wants to use some othe stocks the code will need to be changed as you will need to declare the market name, if the the assets are in different markets. 
The Data has around 1723 rows and 46 columns.

Step 2- Preprocessing
To get returns data use the following code - 

returns = data.pct_change().dropna()
The code gets the returns and drops na values.

I had not done any exploratory data analysis on the portfolio, but one can get covarrince matrix and comparing the values of assets with respect to the market index could be good to start with as it gives the idea of how the portfolio is balacned and how the stocks perform as comapred to market.

I have used covarince values but only to for optimization in later step. 

Step 3-  Assigning Market Steps and intial weights
In this step I assign the market states using the following code

def assign_states(returns, quantiles=(0.33, 0.67)):
    returns = pd.DataFrame(returns)#convert return to dataframe
    avg_ret = returns.mean(axis=1)#get average returns per day
    thresholds = [avg_ret.quantile(q) for q in quantiles]#set thresholds less than 0.33 bear, between 0.33 and 0.67 neutral state
    #lastly greater than 0.67 bull state.
    states = pd.cut(avg_ret, bins=[-np.inf, *thresholds, np.inf], labels=[0, 1, 2])

    return states
states = assign_states(portfolio_returns)
print("States value counts:")
print(states.value_counts())

The code is simple as we take the portfolio returns and assign them states Bear, Bull or Neutral, if it is less than 0.33 than Bull, if between 0.33 and 0.67 Neutral and lastly if greater then 0.67 it is Bear.
The returns are portfolio returns and not just normal stocks returns, to get the portfolio returns, for this we have to first assign intial weights, I have used following code - 

stock_weights = 0.7 / len(stocks)
bond_weights = 0.3 / len(bonds)

weights = {asset: stock_weights if asset in stocks else bond_weights for asset in assets}
weights_array = np.array([weights[asset] for asset in assets])
portfolio_returns = returns.dot(weights_array)

I have given 0.7 to all the stocks and 0.3 to bonds, the intial weights are just 0.7/ number of stocks and 0.3/ number of bonds.

Next, I have performed Markov Chains using the following code -  

def markov_chain(states, threshold=1e-8, max_iter=1000):
    #get current and next possible states
    current_states = states[:-1].reset_index(drop=True)
    next_states = states[1:].reset_index(drop=True)
    #get transition matrix using pd.crosstab, which wwill count the number of times a state transitioned from one to other and then normalize it to get probabilities
    transition_matrix = pd.crosstab(current_states, next_states, normalize='index').to_numpy()
    #intial vector of 1/3
    state_vector = np.ones(transition_matrix.shape[0]) / transition_matrix.shape[0]
    #steady state tells us the time model spends in one state
    #to get state vector we do following steps
    for i in range(max_iter):
        next_vector = state_vector @ transition_matrix #get next vector by multipplying state vector with transtion matrix
        if np.linalg.norm(next_vector - state_vector) < threshold: #if the differnece is less tha above given threshold break from loop and return values
            break
        state_vector = next_vector#else state vector is changed
    return transition_matrix, state_vector

max_iter is set to 1000 but can be changed.  Rather than using loops I have used vectors to get faster calculations. 

In the above code using historical data, we create a transition matrix that captures the probabilities of moving from one state to another, and a steady-state vector is then computed, representing the long-term behavior of the market under these conditions. The iterative process of refining the state vector ensures that our model converges to stable probabilities, even in complex market scenarios. This probabilistic foundation enables the system to anticipate market trends and inform portfolio decisions more effectively.

The results of markov chains lokks like this. - 

Transition Matrix:
[[0.35915493 0.27640845 0.36443662]
 [0.32593857 0.37713311 0.29692833]
 [0.30511464 0.36684303 0.32804233]]
Steady State Distribution: [0.33004067 0.34049971 0.32945961]

The tranistion matris shows the probabilty of a state transitioning to another.

Step 4 - Next I have performed Monte carlo simulations  
To perform Monte carlo simulation we need the current state, transtional probablities and we need the mean and standard deviation of returns of each state. We have transition matrix from Markov chains with the probabilities, to get current states we can get the most recent state from the states variable above and we can get mean and standard deviation by performing basic function on portfolio returns dataframe. 

Once we done with them we run the following code for Monte Carlo simulations - 

def monte_carlo(current_state, transition_matrix, state_stats, n_simulations=10000, n_days=252):
    simulations = []#To store simulation result
    for i in range(n_simulations):#for n simulations
        state = current_state#get currrent state
        daily_returns = []#to store daily returns during simulation
        for j in range(n_days):#for one year
            mean, std = state_stats[state]#get mean and standard deviation for the state
            daily_return = np.random.normal(mean, std)#daily return baised on mean and standard deviation
            daily_returns.append(daily_return)
            state = np.random.choice([0, 1, 2], p=transition_matrix[state])#choose a state based on probabilites
        simulations.append(np.exp(np.cumsum(daily_returns)))#append the cumalitive value of daily returns
    return np.array(simulations)

current_state = states.iloc[-1] #most recent state
simulations = monte_carlo(# do mnte calro simulations
    current_state=current_state,
    transition_matrix=transition_matrix,
    state_stats=state_stats,
    n_simulations=10000,
    n_days=252
)

the code runs 10000 simulations for 252 days as Market is open for that many days in an year. In this the funtion calculates daily returns for 252 days while shifhting to next state based on the transition matrix probabilities. 

This how the first 100 simulations look.

Step 5-Particle filter and Optimization of Portfolio 

#filter will focus on paths that are more consistent
def particle_filter(simulations, values):
    particles = simulations[:, -1] #get the final portfolio values from simulation
    likelihoods = np.exp(-0.5 * ((particles - values.mean()) / values.std()) ** 2)#the likelihood of each particle based on returns
    weights = likelihoods / likelihoods.sum()#normalize
    resampled_indices = np.random.choice(len(particles), size=len(particles), p=weights)#resample based on the weights
    return simulations[resampled_indices], weights

Particle Filters further enhance our portfolio optimization framework by focusing on paths that align more consistently with real-world data. This algorithm begins by extracting the final portfolio values from Monte Carlo simulations and computing the likelihood of each particle based on deviations from the observed data's mean and standard deviation. 

First 100 simulations after particle filter.

After this particles are resampled after being normalized into weights. By taking into consideration both simulated and observed data, particle filters guarantee the consistency and robustness of our portfolio analysis. The precision of dynamic portfolio modifications is greatly improved by this technique, particularly in erratic market conditions.

To use the Particle Filter simulations and weights to optimize the portfolio I have used the following function - 

#funtion to optimize the portfolio
def optimize(expected_returns, covariance_matrix, regularization=0.01):
    def objective(weights):
        p_return = np.dot(weights, expected_returns)#calculate portfolio returns
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))#volatility
        sharpe_ratio = (p_return - risk_free_rate) / p_volatility#calculate sharpe ratio
        penalty = regularization * np.sum(weights**2)#penalize large weights
        return -(sharpe_ratio - penalty)#return sharpe ratio -penalty to minimize
#without penalty the model will not learn to distribute the portfolio
#when I ran the without portfolio it would assign weights only to bonds I have taken as they have steady growth over time and have better returns than most of the stocks with lesser risk and volatility
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]#making sure weights sum is not more than 1
    bounds = [(0, 0.3) for i in expected_returns]  # Enforce diversification

    result = minimize(objective, x0=[1/len(expected_returns)] * len(expected_returns),
                      constraints=constraints, bounds=bounds)#using scipy.optimize.minimize(import) we calculate optimized weights, with pbjective function, bounds, initial weights and constraints as inputs.
    return result.x#return otimized weights

It could be the case that the risk in all of the stocks is low and returns are similarly balanced like in the case of stocks I had taken most of them had low risk decent returns, while others had very low returns. To make sure that model distributes weights in way that the portfolio is versatile and not just have two safe assets, I have used regularization and penalty.Using them the model assigns weight with diversification. the values for regularizatioin and risk_free_rate used to calculate the sharpe ratio can be changed accorsding to need. The bounds set work as constraints on the weight asssignment.

We get the following portfolio as the result - 

Filtered Assets: ['NVDA', 'META', 'NFLX', 'QCOM']
Filtered Weights: [0.3 0.3 0.1 0.3]

Comparing the Portfolio performance - 

As we can see the optimized weights portfolio mostly has better returns as compared to equal weights (my base portfolio) portfolio. At times it  underperforms like in 2018 , 2019, but this could be as the number of assets in optimal assets is very less. what one can do is decrease the bound to 0.1 or 0.15 rather than 0.3. 

Adding Market Sentiments -

To add market sentiments to the model I had used web scrapping code from this github page here. I had tried my own code but since I donot have any API keys avialble for NYT or any other finance pages and yahoo and google scarping where not working correctly, I had used this.  (I had made changes to the original code to make it work for my problem)

In this model I had made some changes to the previous model, Firstly, based on the sentiments states were assigned to the news scrapped. 

If the average sentiment is less than 0.37 it bear, between 0.37 and 0.67 Neutral and is more than 0.67 it is Bull.

Then using the below code for markov chains calulate the transtion matrix and steady state probabilities -

def compute_markov_chain(states):
    unique_states = states.unique() #get unique states
    state_indices = {state: i for i, state in enumerate(unique_states)} #enumerate them
    num_states = len(unique_states)#total states

    transition_counts = np.zeros((num_states, num_states)) #mnake transition from one state to next state as zero
    for (current_state, next_state) in zip(states[:-1], states[1:]):#using both current and next state calculate transition counts
        i, j = state_indices[current_state], state_indices[next_state]
        transition_counts[i, j] += 1

    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)# make transition matrix
    state_vector = np.ones(num_states) / num_states #get state vector
    for i in range(1000): # for x iterations
        next_vector = state_vector @ transition_matrix#get next state
        if np.linalg.norm(next_vector - state_vector) < 1e-8: #if the differnece is not bigger than given threshold break and return matrix and state vector
            break
        state_vector = next_vector#else we have a new state vector

    return transition_matrix, state_vector

The new transition matrix looks like this - 

Transition Matrix:
 [[0.41282051 0.30512821 0.28205128]
 [0.27529412 0.41411765 0.31058824]
 [0.2761194  0.32587065 0.39800995]]
Steady-State Probabilities:
 [0.31950746 0.35014229 0.33035025]

We can see there is shift in probabilities if in Bear state it now clear higher probability that the next state will also be Bear. If we see the diagonal of the matrix it shows that the next state has high probabilty of being the same as current state, as compared to previous transtion matrix.

Using these probabiltie I ran the monte carlo simulation and the optimization function again on the base portfolio with inital weights. We get the new portfolio with following weights -

Filtered Assets: ['MDT', 'ACN', 'NEE', 'AXP']
Filtered Weights: [0.10022125 0.29977875 0.3 0.3]

The portfolio performance with respect to base portfolio looks like this -  

the portfolio performs not only better than the base portfolio but also better than returns portfolio created eralier. 

Concluisions - 

Both of the portfolios had assets with positive correlations with assets in sentiment analysis having more correlation than returns portfolio. 

This project can be used to make an already existing portfolio into otptimized form. 

To achieve optimal results, one can employ both the sentiment and returns-based models in a complementary manner. A sentiment-based model can be used to generate an initial portfolio by leveraging market sentiment data to inform asset allocation decisions. Subsequently, a returns-based model can test and validate the performance of the optimized portfolio by assessing its returns and risk under different market conditions.

The model can be improved by adding data sources and macroeconomic indicators, to further refine the AI model.
