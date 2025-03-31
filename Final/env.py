import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np





#############################
# Environment Wrapper Class #
#############################
class SimEnv:
    def __init__(self, sim, drone_handle, joint_handles, target):
        self.sim = sim
        self.drone_handle = drone_handle
        self.joint_handles = joint_handles
        self.target = target
        self.m = sim.getObjectMatrix(sim.getObject("/base"))
        self.m[3]=0
        self.m[7]=0
        self.m[11]=0
        self.respondables = [sim.getObjectParent(sim.getObjectParent(i)) for i in joint_handles]
        
    def applyForce(self,av):
        self.m = self.sim.getObjectMatrix(self.sim.getObject("/base"))
        self.m[3]=0
        self.m[7]=0
        self.m[11]=0

        hover_force = 9.81 * 0.12*25  # Force required to hover per propeller
        f = [self.sim.multiplyVector(self.m,[0,0,hover_force * av[i]]) for i in range(4)]
        t = [self.sim.multiplyVector(self.m,[0,0,av[i+4]*(1 - (i%2 )* 2/3)]) for i in range(4)]
        for i in range(4):
            self.sim.addForceAndTorque(self.respondables[i],f[i],t[i])  

    def reset(self):
        """
        Resets the simulation by stopping and starting it again.
        Returns the initial state: concatenated position and velocity.
        """
        self.sim.stopSimulation()
        time.sleep(1)  # Allow time for the simulation to stop
        self.sim.startSimulation(self.sim.simulation_stopped)
        time.sleep(0.1)  # Allow time for initialization
        position = self.sim.getObjectPosition(self.drone_handle, -1)
        distance = np.linalg.norm(np.array(position) - np.array(self.target))      
        position = self.sim.getObjectPosition(self.drone_handle, -1)
        velocity = self.sim.getObjectVelocity(self.drone_handle)
        velocity = [item for sublist in velocity for item in sublist]
        state = np.array(position + velocity )  # state is a 6-dimensional vector
        state = np.append(state, distance)
        
        return state

    def step(self, action):
        """
        Applies an action to the drone, steps the simulation one tick,
        and returns (next_state, reward, done, info).
        """
        
        self.sim.step()  # Advance the simulation by one tick
        self.applyForce(action)
        
        # Get updated state information
        position = self.sim.getObjectPosition(self.drone_handle, -1)
        distance = np.linalg.norm(np.array(position) - np.array(self.target))
        velocity = self.sim.getObjectVelocity(self.drone_handle)
        velocity = [item for sublist in velocity for item in sublist]
        next_state = np.array(position + velocity)
        next_state = np.append(next_state, distance)
        
        velocity_penalty = np.linalg.norm(np.array(velocity[0:3]))
        # Reward: negative Euclidean distance to the target   
        reward = -distance  # Main objective: minimize distance to target

        # Reward for staying near the goal
        reward += max(0, 50 * (0.3 - distance))  # Encourages smooth approach

        # Penalize erratic angular velocity (smoothly)
        reward -= 5 * (velocity[5] ** 2)  # Penalize excessive spinning

        up_vector = [self.m[2], self.m[6], self.m[10]]  # Drone's Z-axis

        if up_vector[2] < 0:
            reward -= 50  # Heavy penalty for flipping
            done = 1 

        # Apply a gradual penalty for large distances
        if distance > 10:
            reward -= 2 * (distance - 10)  # Less harsh than -100

        # Prevent crashing, but also reward safe altitude maintenance
        if position[2] < 0.5:
            reward -= 10
        elif position[2] > 1.0:
            reward += 5  # Encourage staying slightly above ground

        # Apply a quadratic velocity penalty for smoother control
        reward -= 0.1 * velocity_penalty ** 2  # Penalize erratic movement


        # Episode termination: if the drone is very close to the target
        done =  distance > 20
        return next_state, reward, done, {}