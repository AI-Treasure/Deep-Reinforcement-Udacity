
# Report

The agents in this project are trained using the Deep Deterministic Policy Gradient (DDPG) algoritm, which is seen as an Actor-Critic approach. Although some researchers think that DDPG is best clssified as DQN method for continous action spaces. This is because in the DDPG approach is that the actions are non stochastic and deterministic. 

But having said this, the Actor and Critic network in DDPG still work in a similar manner compared to other Actor-Critic algorithms. That is, the actor network takes in the current state and outputs an action set for that state. The critic uses this action and the state and returns the estimated Q-value of the state-action pair. This Q value is then used by the actor to evaluate the action.or network to evaluate its choice of action.

Other elements in this approach are :
- Local and target network for both the Actor and the Critic (where two target NN are updated using soft updates)
- Experience replay
- In the actions, noise has been added using the Ornstein-Uhlenbeck process. This in general encourages exploration.

Below we specify the hyperparameters that have been used in the training


|Name|Value|
|---|---:|
|Actor Learning Rate|0.001|
|Critic Learning Rate|0.001|
|Weight Decay|0|
|Gamma|0.99|
|Tau|0.001|
|Buffer Size|100000|
|Batch Size|128|

For the local and target network for the actor, a fully connected layer with 256 nodes, followed by a fully connected layer of 128 nodes has been used, before tuning it down to the action space with a tanh. (Note that this network is fully deterministic.)

For the local and target network for the critic, also a fully connected layer with 256 nodes, followed by a fully connected layer of 128 nodes has been used.

The agents solved the environment in  178 episodes. So after 178 episodes, a collective average reward of 30.0 over the last 100 episodes was obtained.

Below is depicted a picture of the rewards over time.

![plot scores](https://github.com/AI-Treasure/Deep-Reinforcement-Udacity/assets/84537758/21927e15-f393-41a3-9e3b-88cf061f2167)

Of course, many improvements can be made, because this is just scratching the surface.

For instance:

* Further parameter tuning can be done
* Batch normalization could be applied
* Leaky ReLu's could be tested
* Other algorithms like A2C or A3C could be used instead of DDPG
* Prioritized replay could be added
