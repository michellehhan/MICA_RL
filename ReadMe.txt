Baseline - Healthy:
	Random carried weight between 1 to 5
	Target Angle is Fixed at 2.1
	Goal = Reach target angle and hold
	Reward = 10 * sign(change error between current and target angle) + change 
	State dimension[9] = [Joint pos, Velocity, Angle Error,  mus_act[6]]
	Action dimension[6] = muscle excitations	

Exo:
	Random carried weight between 1 to 5
	Target Angle is Fixed at 2.1
	Goal = Reach target angle and hold
	Reward = 10 * sign(change error between current and target angle) + change 
	State dimension[10] = [Joint pos, Velocity, Angle Error,  mus_act[6], Exo_length]
	Action dimension[1] = Exo Actuation
