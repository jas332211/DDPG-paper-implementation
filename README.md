# DDPG-paper-implementation
Implementation of [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) paper

Together with [Stochastic Weighted Average](https://arxiv.org/abs/1803.05407) ([using pytorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)) for a better stability



# Main functions and differences between them:
1)	” DDPG with PyTorch SWA” - using torchcontrib’s SWA version, SWA_start variable is set in timestep units, therefore I set the main loop to run in timesteps aswell! 
2)	“DDPG with PyTorch SWA manual” – using torchcontrib’s SWA version, but this time in manual mode. Enabling myself to set the main loop as well as SWA_start in episode’s units (which is much easier readable)
3)	“DDPG with SWA own impl” – using my own impl of SWA, this version is not finished yet, i met a lot of problems with batch normalization 

# Side functions/classes:
1)Actor_Critic_nets – self-explanatory (these are nets with batch normalization, I’ve also got a set with Layer Normalization so that I wouldn’t need to retrain the model after SWA swap)<br/>
2)Noise – self-explanatory Ornstein–Uhlenbeck process<br/>
3)ReplayBuffer – again, self-explanatory

I’m using mixture of Adam and SWA, setting SWA to 1/5th learning rate of Adam in the moment of swapping, I haven’t yet tested with SGD, since convergences time takes ages

The other problem is optim.bn (DataLoader, model) function which I’m supposed to use at the end of the training after swapping SWA weights to get them normalized, I was unable to connect DataLoader function together with the gym so far


Requirements.txt include the exact Anaconda environment I ran the python on.<br/>
Minimal Requirements.txt includes what I believe should be the minimum to run the program.




# Results & Further improvements: 
The results I was getting from using networks with layer and env normalizations were quite successful: 
 
(As You can see SWA was getting already much better results , and that was only on 350 episodes and using Adam optimizer without batch normalization ) 



There is still a lot of room for improvement and further testing and fine-tunning which will obviously have to be done on proper dataset, some of them include: <br/>
Using different optimizers: Adam/SGD/AdaGrad with different learning rates&decays<br/>
Swapping SWA weights and continuing the training<br/>
Using different Noise<br/>
Changing the size of the buffer so that we only experience the most recent states <br/>
Etc.

