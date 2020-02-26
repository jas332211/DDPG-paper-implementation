env = "Pendulum-v0"
#%matplotlib inline

#Imports : 
import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.optim import Adam,SGD
from torchcontrib.optim import SWA

from ReplayBuffer import ReplayBuffer , Experience
from Actor_Critic_Nets import Actor,Critic
from Noise import OU_Noise

env = gym.make(env)

#Init arguments
num_inputs = env.observation_space.shape[0]
action_space = env.action_space
num_actions = env.action_space.shape[-1]



seed = 0
batch_size = 64
buffer_size = 6000000

gamma = 0.99
tau = 0.001
noise_stddev = 0.2

lr_Actor = 1e-4
lr_Critic = 1e-3
hidden_size = (400,300)  
nr_of_test_cycles = 15 #in episodes

nr_of_timesteps = 120000  #  ~ 600 episodes 
episode_time = 3000 # limit of timesteps per episode
warmup_value= 100

SWA_freq =  100
SWA_lr_actor= 1/5 * lr_Actor
SWA_lr_critic = 1/5*lr_Critic
SWA_start = 80000 #timesteps

env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

testing_model = "actor_target" # we can test either using actor or actor_target 

if (torch.cuda.is_available()) : 
    device = "cuda"
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else :
    device = "cpu"

    
test_rewards_SWA , rewards_list, mean_test_rewards, episode_list , test_rewards, test_episode_list = [], [], [], [], [], []



#DDPG ALOGRITHM : 

#1)RANDOMLY Initialize Critic network and actor network 
actor = Actor(hidden_size, num_inputs, action_space).to(device)
critic = Critic(hidden_size, num_inputs, action_space).to(device)


#2)Initialize targer actor/critic with same weights as actor/critic
actor_target = Actor(hidden_size, num_inputs, action_space).to(device)
critic_target = Critic(hidden_size, num_inputs, action_space).to(device)

for target_parameter, parameter in zip (actor_target.parameters(),actor.parameters()):
    target_parameter.data.copy_(parameter.data)
    
for target_parameter, parameter in zip (critic_target.parameters(),critic.parameters()):
    target_parameter.data.copy_(parameter.data)




#Optimizers*
actor_optimizer2 = Adam(actor.parameters(),
                            lr=lr_Actor)  
critic_optimizer2 = Adam(critic.parameters(),
                                lr=lr_Critic,
                                )
#implement SWA  
actor_optimizer = SWA(actor_optimizer2,
                        swa_lr= SWA_lr_actor,
                        swa_freq = SWA_freq,
                        swa_start=SWA_start) 

critic_optimizer = SWA(critic_optimizer2,
                        swa_lr= SWA_lr_critic,
                        swa_freq = SWA_freq,
                        swa_start=SWA_start) 



#3)Initialize replay Buffer R
memory = ReplayBuffer(buffer_size)

#4)*****INSTEAD OF FOR EPISODE LOOP WE RUN FOR OVERALL

SWA_freq_count = 0
episode = 0
timestep = 0
while (timestep <nr_of_timesteps) :
    episode +=1 
    episode_return = 0
    curr_t = 0
    print("episode : ",episode)
    
    
    #4.1)Initialize a random Process OU
    ou_noise = OU_Noise(mu=np.zeros(num_actions
), sigma=float(noise_stddev) * np.ones(num_actions
))

    #4.2)Receive initial observation of stat
    state = torch.Tensor([env.reset()]).to(device)

    #4.3)for loop , t=1 -->T :
    while (True and curr_t<episode_time) :  #Just for clarity , it finishes if either we run out of moves or time 
        curr_t+=1
        timestep+=1
        #4.3.1)select action chosen by actor and add OUth random process to the choice 
        actor.eval()  # Sets the actor in evaluation mode
        action = actor(state)
        actor.train()  # Sets the actor in training mode
        noise = torch.Tensor(ou_noise.sample_noise()).to(device)
        action = action.data  + noise
        action = action.clamp(action_space.low[0], action_space.high[0]) 
        #^^ to make sure that after adding noise to action we don't end-up with action outside of action space

        
        #4.3.2)Execute selected action and observe : reward,new state
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        episode_return += reward

        
        #4.3.3)Store previous state,action,reward,and new state in Buffer R
        terminal = torch.Tensor([done]).to(device)
        reward = torch.Tensor([reward]).to(device)
        next_state = torch.Tensor([next_state]).to(device)

        memory.push(state, action, next_state, reward,terminal)
        state = next_state

        #4.3.4)Sample a random minibatch of N transitions from Buffer R
        if len(memory)> warmup_value:
            Experiences = memory.sample(batch_size)
            batch = Experience(*zip(*Experiences))

            states = torch.cat(batch.state).to(device)
            terminals = torch.cat(batch.terminal).to(device)
            actions = torch.cat(batch.action).to(device)
            rewards = torch.cat(batch.reward).to(device)
            next_states = torch.cat(batch.next_state).to(device)


            #4.3.5) calculate batch of Q values dependent on target networks 
            next_actions = actor_target(next_states)
            next_state_action_values = critic_target(next_states, next_actions.detach())
            
            rewards = rewards.unsqueeze(1)
            terminals = terminals.unsqueeze(1)
            Q_values_by_target = rewards + (1.0 - terminals) * gamma * next_state_action_values
            
            
            #4.3.6)Update critic using mean squared error loss between Q_values calc on target vs those by non-target
            critic_optimizer.zero_grad()
            
            Q_values_by_non_target = critic(states, actions)
            critic_loss_function = F.mse_loss(Q_values_by_non_target, Q_values_by_target.detach())
            
            critic_loss_function.backward()
            critic_optimizer.step()

            
            #4.3.7)Update actor using Policy Gradient calculated by non_target networks
            actor_optimizer.zero_grad()
            
            policy_loss = -critic(states, actor(states))
            policy_loss = policy_loss.mean()
            
            policy_loss.backward()
            actor_optimizer.step()

            #4.3.8)Update both targer function by tau
            for p_target , p in zip(actor_target.parameters(),actor.parameters()):
                p_target.data.copy_(p_target.data * (1.0 - tau) +  p.data * tau)
                
            for p_target , p in zip(critic_target.parameters(),critic.parameters()):
                p_target.data.copy_(p_target.data * (1.0 - tau) +  p.data * tau)

        if done:
            break
            
    rewards_list.append(episode_return)
    episode_list.append(episode)



    
    
    


#########################    TESTING BEFORE SWA WEIGHT SWAP   #####################################

if (testing_model == "actor_target") :
    testing_model = actor_target
else : 
    testing_model = actor
    


for episode in range (nr_of_test_cycles) : 
    
    ou_noise = OU_Noise(mu=np.zeros(num_actions
), sigma=float(noise_stddev) * np.ones(num_actions
))
    test_episode_return = 0
    state = torch.Tensor([env.reset()]).to(device)

    print("episode : ",episode)

    while True : 
        testing_model.eval()  # Sets the actor in evaluation mode
        action = testing_model(state)
        testing_model.train()  # Sets the actor in training mode
        noise = torch.Tensor(ou_noise.sample_noise()).to(device)
        action = action.data  + noise
        action = action.clamp(action_space.low[0], action_space.high[0])

        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

        test_episode_return += reward
        next_state = torch.Tensor([next_state]).to(device)
        state = next_state

        if done:
            break

    #append lists so we can graph them later 
    test_episode_list.append(episode)
    test_rewards.append(test_episode_return)





##################   Testing after SWA swap    ########################

actor_optimizer.swap_swa_sgd()
critic_optimizer.swap_swa_sgd()


#since we're using batchNorm we've to pass normalization through our model : 

#opt.bn_update(train_loader = , model=Actor)

for episode in range (nr_of_test_cycles) : 
    
    test_episode_return = 0
    ou_noise = OU_Noise(mu=np.zeros(num_actions
), sigma=float(noise_stddev) * np.ones(num_actions
))
    state = torch.Tensor([env.reset()]).to(device)

    print("episode : ",episode)

    while True : 
        actor.eval()  # Sets the actor in evaluation mode
        action = actor(state)
        actor.train()  # Sets the actor in training mode
        noise = torch.Tensor(ou_noise.sample_noise()).to(device)
        action = action.data  + noise
        action = action.clamp(action_space.low[0], action_space.high[0])

        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

        test_episode_return += reward
        next_state = torch.Tensor([next_state]).to(device)
        state = next_state

        if done:
            break

    #append lists so we can graph them later 
    test_rewards_SWA.append(test_episode_return)





env.close()
#plot the results 

plt.subplot(2,1,1)
plt.plot(episode_list,rewards_list)
plt.xlabel("episodes")
plt.ylabel("training rewards")
plt.title("training rewards , no SWA swaps inbetween")
plt.legend(["rewards"])

plt.subplot(2,1,2)
plt.plot(test_episode_list,test_rewards)
plt.plot(test_episode_list,test_rewards_SWA)
plt.xlabel("test episodes")
plt.ylabel("test rewards")
plt.legend(["test_rewards","SWA_test_rewards"])

plt.show()
