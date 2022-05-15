import numpy as np
import pickle
import random
import pandas as pd
from tqdm import tqdm

corr = None
ifTest = False
Ztr = {}
Zte = {}


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


# total number of States and Actions
n_states =100
n_actions = 3
# np.random.seed(0)
# initial Q-table
Q = np.zeros([n_states, n_actions])
epsilons = []
cum_rewards=[]

# learning rate
alpha = 0.1

# discount factor
gamma = 0.99
# e-greedy exploitation
epsilon = 0.9
epsilon_decay = 0.01
epsilon_final = 0.001

# training parameters
n_episodes = 2000
n_steps = 100

def getState(state, epsilon):
    """this function returns state based on the value of random number generated"""
    global Q
    p = np.random.uniform(0,1)
    
    action =None
    if p>epsilon:
        rand_values = Q[state]
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
    
    for i in tqdm(range(n_episodes)):
        # randomly generaating states between 0 and 100
        state = random.randint(0,99)
        # initial cumulative reward set
        cum_reward =0
        # initial price of bond
        inventory=[]
        price =100
        balance = 0
        bondbal = 0
        networth = 0
        
        for j in range(n_steps):
            # performing n_steps within each iteration
            p = np.random.rand()
            
            action = None
            if p>epsilon:
                rand_values = Q[state]
                action = np.argmax(rand_values)-1
            else:
                action = np.random.randint(n_actions)-1
            
            picc, nxt_state = returnNext((state//10,state%10), action)
            
            new_state =nxt_state[0]*10+nxt_state[1]
            
            old_networth = networth
            
            reward =0
            price+=picc
            if action==1:
                inventory.append(price)
                balance-=10*price
                bondbal+=10
            elif action ==-1 and len(inventory)>0:
                balance+=10*price
                bondbal-=10
            
            networth = balance + bondbal*price
            delta = networth - old_networth
            reward =delta
#             reward = max(delta,0)
            
            Q[state][action+1] = ((1-alpha)*Q[state][action+1]) + alpha*(reward + gamma*np.max(Q[new_state]))
            
            cum_reward+=reward
            state = new_state
        
        
        if epsilon > epsilon_final:
                epsilon*=(1-epsilon_decay)
                epsilons.append(epsilon)
        cum_rewards.append(cum_reward)
        
# This you need to write after training your MDP, this should just take in the state and return the Action you would perform
# Please do not make use of the returnNext function inside here, that would defeat the purpose of training the model, as it would be known to you!

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
        # if i%(iter//10)==0:
        #     print('Your Networth has went from 0 to ',networth)
        print('Your Networth has went from 0 to ',networth)
        i+=1


ftr = input('Train : Filename to read?')
fte = input('Test : Filename to read from?')

# ftr="./train.pic"
# fte="./test.pic"

with open(ftr,'rb') as f:
    Ztr = pickle.load(f)

with open(fte,'rb') as f:
    Zte = pickle.load(f)

with open('correlations.npy','rb') as f:
    corr = np.load(f)

ifTest  = False
Train()
# Q =np.load('./Q.npy')
ifTest = True
mainRun(100000)
