import numpy as np
import pickle
import random
import pandas as pd
corr = None
ifTest = False
Ztr = {}
Zte = {}


# total number of States and Actions
n_states =100
n_actions = 3
#np.random.seed(0)


# initial Q-table
Q = np.zeros([n_states, n_actions])
epsilons = []

# learning rate
alpha = 0.99

# discount factor
gamma = 1.0

# e-greedy exploitation
epsilon = 0.9
epsilon_decay = 0.01
epsilon_final = 0.001


# training parameters
n_episodes = 200
n_steps = 100


def returnNext(state, action):
    # Returns (priceChange, nextState) as a tuple
    #nextState is actually a 2-tuple representing the state
    a,b = state
    L = []
    if ifTest:
        L = Zte[(a,b)]
    else:
        L = Ztr[(a,b)]
    r = L[random.randint(0,len(L)-1)]
    pricc = None
    if action==-1:
        if r[0]==0:
            pricc =  r[4]
        elif r[0]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    elif action==1:
        if r[2]==0:
            pricc =  r[4]
        elif r[2]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    else:
        if r[1]==0:
            pricc =  r[4]
        elif r[1]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    num = 3*(10*a + b) + action + 1

    F = corr[num,:]
    ch = np.random.choice(np.arange(0,100),p = list(F))
    ca = ch // 10
    cb = ch % 10
    return (pricc,(ca,cb))

def getState(state, epsilon):
    global Q
    p = np.random.rand()
    
    action =None
    if p>epsilon:
        rand_values = Q[state]+ np.random.rand(1,n_actions)/1000
        action = np.argmax(rand_values)-1

    else:
        action = np.random.randint(n_actions)-1
        
    return action

def getMinIndex_Value(l):
    min_value = min(l)
    min_index = l.index(min_value)
    
    return min_value, min_index

# Train your Markov Decision Process, make use of the returnNext function to model and act on the priceChanges dependence on the states and actions
def Train():
    
    global n_episodes, n_steps, epsilon, epsilon_decay, epsilon_final, gamma, alpha, epsilons, Q
    
    for i in range(n_episodes):
        state =0
        cum_reward =0
        done =False
        
        inventory = []
        price =100
        
        for j in range(n_steps):
            p = np.random.rand()
            
            action = None
            if p>epsilon:
                rand_values = Q[state]+ np.random.rand(1,n_actions)/1000
                action = np.argmax(rand_values)-1
            
            else:
                action = np.random.randint(n_actions)-1
            
            
            
            if epsilon > epsilon_final:
                epsilon*=(1-epsilon_decay)
            
            picc, nxt_state = returnNext((state//10,state%10), action)
            
            new_state =nxt_state[0]*10+nxt_state[1]
                                         
            reward =0
            price+=picc
            if action==1:
                inventory.append(price)
            elif action ==-1 and len(inventory)>0:
                bought_price, index = getMinIndex_Value(inventory)
                del inventory[index]
                reward = max(price-bought_price,0)
            
            Q[state][action] = ((1-alpha)*Q[state][action]) + alpha*(reward + gamma*np.max(Q[new_state]))
            
            cum_reward+=reward
            state = new_state
            
            
#             print(state, action, price, reward, inventory)

def Run(state):
    global epsilon
    s = state[0]*10+ state[1]%10
    action =getState(s, epsilon)
    return action

# This is the main function, you don't need to tamper with it!
def mainRun(iter = 1000):
    initstate = (random.randint(0,9),random.randint(0,9))
    i = 0
    initprice = 100.00
    price = initprice
    balance = 0
    bondbal = 0
    networth = 0
    st = initstate
    while i < iter:
        act = Run(st)
        pricCh, ns = returnNext(st,act)
        price += pricCh
        if act==1:
            balance -= 10*price
            bondbal += 10
        elif act==-1:
            balance += 10*price
            bondbal -= 10
        
        networth = balance + bondbal*price
        st = ns
        print('Your Networth has went from 0 to ',networth)
        i+=1


ftr = input('Train : Filename to read?')
fte = input('Test : Filename to read from?')


with open(ftr,'rb') as f:
    Ztr = pickle.load(f)

with open(fte,'rb') as f:
    Zte = pickle.load(f)

with open('correlations.npy','rb') as f:
    corr = np.load(f)

ifTest  = False
Train()
ifTest = True
mainRun()
