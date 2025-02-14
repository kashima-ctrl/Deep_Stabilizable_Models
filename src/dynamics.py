import torch.nn as nn
import torch as th
import numpy as np
from torch.utils.data import TensorDataset

class Dynamics(nn.Module):
    def __init__(self, state_dim:int, action_dim:int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def make_dataset(self,num:int,isRandom:bool)->TensorDataset:
        return None
    
class VanDerPol(Dynamics):
    def __init__(self, mu: float = 0.3, tau: float = 0.02) -> None:
        """
        Initializes the Van der Pol oscillator dynamics.

        Parameters:
            mu (float): The nonlinearity parameter that controls the strength of
                        the oscillation. Larger values increase oscillatory behavior.
            tau (float): The time step between state updates.
        """
        super().__init__(state_dim=2, action_dim=1)
        self.mu = mu
        self.tau = tau  # time interval for state updates

    def state_dot(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the state (state_dot) based on the current
        state and action input using the Van der Pol equations.

        Parameters:
            state (np.ndarray): The current state of shape (N, 2), where the first
                                column is position `x`, and the second is velocity `x_dot`.
            action (np.ndarray): Control input, though it typically does not affect
                                 Van der Pol dynamics directly (set to 0 for now).

        Returns:
            np.ndarray: The time derivative of the state, state_dot, with shape (N, 2).
        """
        x = state[:, 0]
        x_dot = state[:, 1]
        u = action.flatten()
        # Van der Pol oscillator equations
        x_ddot = - x + self.mu * (1 - x_dot**2) * x_dot  + u
        return np.column_stack((x_dot, x_ddot))

    def plot(self,u=0):
        """
        Plots the phase portrait of the Van der Pol oscillator.
        """
        import matplotlib.pyplot as plt

        # Define the ranges for `x` and `x_dot` and create a grid
        x = np.linspace(-3, 3, 200)
        x_dot = np.linspace(-3, 3, 200)
        x, x_dot = np.meshgrid(x, x_dot)
        x = x.flatten()
        x_dot = x_dot.flatten()

        # Generate state derivatives for each point in the grid
        state = np.column_stack((x, x_dot))
        action = np.ones_like(x)*u  # Action is zero for Van der Pol dynamics
        state_dot = self.state_dot(state=state, action=action)

        # Compute the speed of state change for color mapping
        dx, dx_dot = state_dot[:, 0], state_dot[:, 1]
        speed = np.sqrt(dx**2 + dx_dot**2)

        # Plot the phase portrait using a streamplot
        plt.figure(figsize=(8, 8))
        plt.streamplot(
            x.reshape(200, 200), x_dot.reshape(200, 200),
            dx.reshape(200, 200), dx_dot.reshape(200, 200),
            color=speed.reshape(200, 200), cmap='autumn', arrowsize=5, linewidth=2.5
        )

        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$\dot{x}$", fontsize=18)
        plt.title("Van der Pol Oscillator Phase Portrait", fontsize=20)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()    

    def make_dataset(self, num=11, isRandom=True) -> TensorDataset:
        """
        Creates a dataset of state-action pairs and their derivatives for the Van der Pol oscillator.

        Parameters:
            num (int): Number of samples to generate along each dimension.
            isRandom (bool): If True, sample randomly within the range; if False, sample linearly.

        Returns:
            TensorDataset: A dataset containing input states and their derivatives.
        """
        # Define the range for the states `x` and `x_dot`
        x_range = 3.0
        u_range = 2.0
        # Randomly or linearly sample initial conditions for `x` and `x_dot`
        if isRandom:
            x = np.random.uniform(-x_range, x_range, num)
            x_dot = np.random.uniform(-x_range, x_range, num)
            u = np.random.uniform(0, u_range, 11)
        else:
            x = np.linspace(-x_range, x_range, num)
            x_dot = np.linspace(-x_range, x_range, num)
            #u = np.linspace(0, u_range, 2)
            u = np.linspace(-u_range, u_range, 11)


        # Generate a grid for all combinations of `x` and `x_dot`
        x, x_dot, u = np.meshgrid(x, x_dot, u)
        state = np.array([x, x_dot]).reshape(2, -1).T
        action = np.array([u]).reshape(1, -1).T
        # Convert state and action to PyTorch tensors
        input_data = th.from_numpy(np.hstack((state, action)).astype(np.float32)).clone()

        # Compute state derivatives
        output_data = self.state_dot(state, action)
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()

        # Return the dataset
        dataset = TensorDataset(input_data, output_data)
        return dataset

class CartPole(Dynamics):  
    def __init__(self) -> None:
        super().__init__(state_dim=4, action_dim=1)     
        """
        OpenAI gymnasium CartPole-v1
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        """
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 3.0
        self.tau = 0.02  # seconds between state updates

        self.x_max = 1.0
        self.x_dot_max = 1.0
        self.theta_max = np.pi/6
        self.theta_dot_max = 1.0


    def state_dot(self, state:np.ndarray, action:np.ndarray):
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            action + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return  np.column_stack((x_dot, xacc, theta_dot, thetaacc))
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        theta = np.linspace(-self.theta_max, self.theta_max, 200)
        theta_dot = np.linspace(-self.theta_dot_max, self.theta_dot_max, 200)
       
        theta, theta_dot = np.meshgrid(theta,theta_dot)
        theta = theta.flatten()
        theta_dot = theta_dot.flatten()
        x, x_dot,  u= np.zeros_like(theta), np.ones_like(theta)*0, np.ones_like(theta)*0
        state = np.column_stack((x, x_dot, theta, theta_dot))
       
        state_dot = self.state_dot(state=state,action=u)
        dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
        speed = np.sqrt(dtheta**2 + dtheta_dot**2)
        
        plt.figure(figsize=(8,8))
        plt.streamplot(theta.reshape(200,200),
                       theta_dot.reshape(200,200),
                       dtheta.reshape(200,200),
                       dtheta_dot.reshape(200,200),
                       color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

        plt.xlabel(r"$\theta$",fontsize=18)
        plt.ylabel(r"$\dot\theta$",fontsize=18)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        
        plt.show()

    def make_dataset(self,num=21, isRandom=True)->TensorDataset:
        x = np.linspace(-self.x_max, self.x_max, 5)
        x_dot = np.linspace(-self.x_dot_max, self.x_dot_max, 5)
        if isRandom:
            theta = np.random.uniform(-self.theta_max, self.theta_max, num)
            theta_dot = np.random.uniform(-self.theta_dot_max, self.theta_dot_max, num)    
        else:    
            theta = np.linspace(-self.theta_max, self.theta_max, num)
            theta_dot = np.linspace(-self.theta_dot_max, self.theta_dot_max, num)
        u = np.linspace(-self.force_mag,self.force_mag,11)

        x, x_dot, theta, theta_dot, u = np.meshgrid(x,x_dot,theta,theta_dot,u)
        state_action = np.array([x, x_dot, theta, theta_dot, u]).reshape(5,-1).T
        input_data = th.from_numpy(state_action.astype(np.float32)).clone()
        output_data = self.state_dot(state_action[:,:4],state_action[:,4])
 
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()
        dataset = TensorDataset(input_data, output_data)
        return dataset
    

    
class Pendulum(Dynamics):
    """OpenAI gymnasium Pendulum-v1"""
    def __init__(self) -> None:
        super().__init__(state_dim=2, action_dim=1)
        self.dt = 0.05
        self.g = 0.98
        self.m = 1.0
        self.l = 1.0

        self.x1_min=-3
        self.x1_max= 3 
        self.x2_min=-3, 
        self.x2_max= 3
        self.u_min = 2
        self.u_max =-2

    def x1_dot(self, x1:np.ndarray, x2:np.ndarray, u:np.ndarray):
        return x2 
    
    def x2_dot(self, x1:np.ndarray, x2:np.ndarray, u:np.ndarray):
        return (3 * self.g / (2 * self.l) * np.sin(x1) + 3.0 / (self.m * self.l**2) * u)

    def make_dataset(self,num=11,isRandom=False)->TensorDataset:    
        tmp_x1 = np.linspace(self.x1_min,self.x1_max,num)
        tmp_x2 = np.linspace(self.x2_min,self.x2_max,num)
        tmp_u = np.linspace(self.u_min,self.u_max,num)
        grid_x1, grid_x2, grid_u = np.meshgrid(tmp_x1,tmp_x2,tmp_u)
        x1 = grid_x1.ravel()
        x2 = grid_x2.ravel()
        u = grid_u.ravel()
        X1 = th.from_numpy(x1.astype(np.float32)).clone()
        X2 = th.from_numpy(x2.astype(np.float32)).clone()
        U = th.from_numpy(u.astype(np.float32)).clone()
        input_data = th.stack([X1, X2, U], 1)
        dx1 = self.x1_dot(x1,x2,u)
        dx2 = self.x2_dot(x1,x2,u)    
        DX1 = th.from_numpy(dx1.astype(np.float32)).clone()
        DX2 = th.from_numpy(dx2.astype(np.float32)).clone()
        output_data = th.stack([DX1,DX2],1)
        dataset = TensorDataset(input_data, output_data)
        return dataset
    
    def plot(self):
        import matplotlib.pyplot as plt
        x1 = np.linspace(self.x1_min, self.x1_max, 200)
        x2 = np.linspace(self.x2_min, self.x2_max, 200)
       
        x, y = np.meshgrid(x1,x2)
        u = 0
       
        dx1 = self.x1_dot(x,y,u)
        dx2 = self.x2_dot(x,y,u)
        speed = np.sqrt(dx1**2 + dx2**2)
        
        plt.figure(figsize=(8,8))
        plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

        plt.xlabel("x_1",fontsize=18)
        plt.ylabel("x_2",fontsize=18)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        
        plt.show()

class BenchmarkExample(Dynamics):
    def __init__(self, tau: float = 0.01) -> None:
        """
        Initializes the BenchmarkExample dynamics.

        Parameters:
            tau (float): The time step between state updates.
        """
        super().__init__(state_dim=3, action_dim=1)
        self.tau = tau  # time interval for state updates

    def state_dot(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the state (state_dot) based on the current
        state and action input using the specified equations.

        Parameters:
            state (np.ndarray): The current state of shape (N, 3), where columns represent x1, x2, x3.
            action (np.ndarray): Control input with shape (N, 1).

        Returns:
            np.ndarray: The time derivative of the state, state_dot, with shape (N, 3).
        """
        x1, x2, x3 = state[:, 0], state[:, 1], state[:, 2]
        u = action.flatten()

        # Define the dynamics equations
        x1_dot = u
        x2_dot = -x2 - 3 * x1 * x2**2
        x3_dot = -2 * x1

        return np.column_stack((x1_dot, x2_dot, x3_dot))

    def observation_model(self, state: np.ndarray) -> np.ndarray:
        """
        Observation model for the system. Returns the observed variable y = x1.

        Parameters:
            state (np.ndarray): The current state of shape (N, 3).

        Returns:
            np.ndarray: Observation y with shape (N, 1).
        """
        return state[:, [0]]  # y = x1

    def make_dataset(self, num=11, isRandom=True) -> TensorDataset:
        """
        Creates a dataset of state-action pairs and their derivatives for the system.

        Parameters:
            num (int): Number of samples to generate along each dimension.
            isRandom (bool): If True, sample randomly within the range; if False, sample linearly.

        Returns:
            TensorDataset: A dataset containing input states and their derivatives and observations.
        """
        # Define the range for states and control input
        x_range = 1.0
        u_range = 1.0

        # Sample initial conditions for `x1`, `x2`, `x3`, and `u`
        if isRandom:
            x1 = np.random.uniform(-x_range, x_range, num)
            x2 = np.random.uniform(-x_range, x_range, num)
            x3 = np.random.uniform(-x_range, x_range, num)
            u = np.random.uniform(-u_range, u_range, num)
        else:
            x1 = np.linspace(-x_range, x_range, 11)
            x2 = np.linspace(-x_range, x_range, num)
            x3 = np.linspace(-x_range, x_range, num)
            u = np.linspace(-u_range, u_range, 11)

        # Create a grid for all combinations of `x1`, `x2`, `x3`, and `u`
        x1, x2, x3, u = np.meshgrid(x1, x2, x3, u)
        state = np.array([x1, x2, x3]).reshape(3, -1).T
        action = np.array([u]).reshape(1, -1).T

        # Convert state and action to PyTorch tensors
        input_data = th.from_numpy(np.hstack((state, action)).astype(np.float32)).clone()

        # Compute state derivatives and observation
        state_dot = self.state_dot(state, action)
        y = self.observation_model(state)
        output_data = th.from_numpy(np.hstack((state_dot, y)).astype(np.float32)).clone()

        # Return the dataset
        dataset = TensorDataset(input_data, output_data)
        return dataset

    def plot(self, u=0):
        """
        Plots x2 and x3 dynamics over a range of initial conditions.
        """
        import matplotlib.pyplot as plt

        # Define the ranges for `x1` and `x2` and create a grid
        x2 = np.linspace(-3, 3, 200)
        x3 = np.linspace(-3, 3, 200)
        x2, x3 = np.meshgrid(x2, x3)
        x2 = x2.flatten()
        x3 = x3.flatten()

        # Set x3 to a constant for plotting, and define action `u`
        x1 = np.zeros_like(x2)  # For plotting purposes, set x3 initially to zero
        state = np.column_stack((x1, x2, x3))
        action = np.ones_like(x1) * u

        # Generate state derivatives for x2 and x3
        state_dot = self.state_dot(state=state, action=action)
        dx2, dx3 = state_dot[:, 1], state_dot[:, 2]

        # Plot the phase portrait for x2 and x3
        plt.figure(figsize=(8, 8))
        plt.streamplot(
            x2.reshape(200, 200), x3.reshape(200, 200),
            dx2.reshape(200, 200), dx3.reshape(200, 200),
            color=np.sqrt(dx2**2 + dx3**2).reshape(200, 200), cmap='autumn', arrowsize=5, linewidth=2.5
        )

        plt.xlabel(r"$x_2$", fontsize=18)
        plt.ylabel(r"$x_3$", fontsize=18)
        plt.title("Benchmark Example Phase Portrait", fontsize=20)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()

class HJI_Example(Dynamics):
    def __init__(self, tau= 0.01) -> None:
        """
        Initializes the 1-dimensional HJI example system with x_dot = x^2 + u.
        """
        super().__init__(state_dim=1, action_dim=1)
        self.tau = tau

    def state_dot(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the state (x_dot) for the system.

        Parameters:
            state (np.ndarray): The current state with shape (N, 1).
            action (np.ndarray): The control input `u` with shape (N, 1).

        Returns:
            np.ndarray: The time derivative of the state, x_dot.
        """
        x = state.flatten()
        u = action.flatten()
        x_dot = x**2 + u
        return np.column_stack((x_dot,))

    def plot(self, u=0):
        """
        Plots the phase portrait of the system by showing x vs x_dot.
        """
        import matplotlib.pyplot as plt
        x = np.linspace(-3, 3, 100)
        state = np.column_stack((x,))
        action = np.ones_like(x) * u  # Set constant control input
        x_dot = self.state_dot(state=state, action=action)

        plt.figure(figsize=(8, 6))
        plt.plot(x, x_dot, label=r"$\dot{x} = x^2 + u$")
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$\dot{x}$", fontsize=18)
        plt.title("HJI Example Phase Portrait", fontsize=20)
        plt.grid()
        plt.legend()
        plt.show()

    def make_dataset(self, num=11, isRandom=True) -> TensorDataset:
        """
        Creates a dataset of state-action pairs and their derivatives for the HJI example.

        Parameters:
            num (int): Number of samples to generate.
            isRandom (bool): If True, samples randomly; otherwise, samples linearly.

        Returns:
            TensorDataset: A dataset with input states and their derivatives.
        """
        x_range = 3.5
        u_range = 3

        if isRandom:
            x = np.random.uniform(-x_range, x_range, num)
            u = np.random.uniform(-u_range, u_range, num)
        else:
            x = np.linspace(-x_range, x_range, num)
            u = np.linspace(-u_range, u_range, 11)

        x, u = np.meshgrid(x,u)
        state = x.reshape(-1,1)
        action = u.reshape(-1,1)
        
        input_data = th.from_numpy(np.hstack((state, action)).astype(np.float32)).clone()

        output_data = self.state_dot(state, action)
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()

        dataset = TensorDataset(input_data, output_data)
        return dataset    
    
class SSG_Example(Dynamics):
    def __init__(self) -> None:
        """
        Initializes the 2D dynamical system:
        x1_dot = -x1 + 5*x2
        x2_dot = -x1^3 - x2 + u
        """
        super().__init__(state_dim=2, action_dim=1)

    def state_dot(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative of the state.

        Parameters:
            state (np.ndarray): Current state (N, 2), where the first column is x1, the second is x2.
            action (np.ndarray): Control input u (N, 1).

        Returns:
            np.ndarray: Time derivative of the state (N, 2).
        """
        x1 = state[:, 0]
        x2 = state[:, 1]
        u = action.flatten()

        x1_dot = -x1 + 5 * x2
        x2_dot = -x1**3 - x2 + u

        return np.column_stack((x1_dot, x2_dot))

    def plot(self, u=0):
        """
        Plots the phase portrait of the system in the x1-x2 plane.
        """
        import matplotlib.pyplot as plt
        x1 = np.linspace(-3, 3, 100)
        x2 = np.linspace(-3, 3, 100)
        x1, x2 = np.meshgrid(x1, x2)
        x1 = x1.flatten()
        x2 = x2.flatten()

        state = np.column_stack((x1, x2))
        action = np.ones_like(x1) * u  # Constant control input
        state_dot = self.state_dot(state=state, action=action)

        dx1, dx2 = state_dot[:, 0], state_dot[:, 1]
        speed = np.sqrt(dx1**2 + dx2**2)

        plt.figure(figsize=(8, 8))
        plt.streamplot(
            x1.reshape(100, 100), x2.reshape(100, 100),
            dx1.reshape(100, 100), dx2.reshape(100, 100),
            color=speed.reshape(100, 100), cmap="autumn", arrowsize=1.5, linewidth=1.5
        )
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("SSG Example Phase Portrait", fontsize=18)
        plt.tight_layout()
        plt.show()

    def make_dataset(self, num=11, isRandom=True) -> TensorDataset:
        """
        Creates a dataset of state-action pairs and their derivatives.

        Parameters:
            num (int): Number of samples to generate.
            isRandom (bool): If True, samples randomly; otherwise, samples linearly.

        Returns:
            TensorDataset: A dataset containing input states and their derivatives.
        """
        x_range = 3.0
        u_range = 3.0

        if isRandom:
            x1 = np.random.uniform(-x_range, x_range, num)
            x2 = np.random.uniform(-x_range, x_range, num)
            u = np.random.uniform(-u_range, u_range, num)
        else:
            x1 = np.linspace(-x_range, x_range, num)
            x2 = np.linspace(-x_range, x_range, num)
            u = np.linspace(-u_range, u_range, num)

        x1, x2, u = np.meshgrid(x1, x2, u)
        state = np.column_stack((x1.flatten(), x2.flatten()))
        action = u.flatten().reshape(-1, 1)

        input_data = th.from_numpy(np.hstack((state, action)).astype(np.float32)).clone()
        output_data = self.state_dot(state, action)
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()

        dataset = TensorDataset(input_data, output_data)
        return dataset
    
    def make_ssdataset(self, num=11, isRandom = True, isPlot = False):
        from scipy.optimize import fsolve
         # Define the equation for x2
        def equation(x2, u): return 125 * x2**3 + x2 - u

        # Define the range of u values
        u_values = np.linspace(-3, 3, num)  # Example: u values from -2 to 2
        x2_solutions = []

        # Solve for x2 for each u
        for u in u_values:
            # Use fsolve to find the root of the equation for each u
            x2_sol = fsolve(equation, x0=0.1, args=(u,))[0]  # Initial guess x0 = 0.1
            x2_solutions.append(x2_sol)

        # Compute x1 for each x2
        x1_solutions = [5 * x2 for x2 in x2_solutions]

        # Convert solutions to arrays for plotting
        x2_solutions = np.array(x2_solutions)
        x1_solutions = np.array(x1_solutions)
        if isPlot:
            import matplotlib.pyplot as plt
            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.plot(u_values, x1_solutions, label=r"$x_1 = 5x_2$", color='blue')
            plt.plot(u_values, x2_solutions, label=r"$x_2$", color='orange')
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            plt.xlabel(r"$u$", fontsize=14)
            plt.ylabel(r"Terminal state", fontsize=14)
            plt.title(r"Terminal States $(x_1, x_2)$ vs $u$", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid()
            plt.show()
        
        output_data =  np.column_stack((x1_solutions.flatten(), x2_solutions.flatten()))
        output_data = th.from_numpy(output_data.astype(np.float32)).clone()
        input_data = th.from_numpy(u_values.astype(np.float32)).clone()

        dataset = TensorDataset(input_data, output_data)
        return dataset
    
if __name__ == "__main__":
    #p = Pendulum()
    #p.plot()
    #p.make_dataset()
    #cp = VanDerPol()
    #cp.plot(u=0)
    #cp.make_dataset()
    #bm = BenchmarkExample()
    #bm.plot()
    #hjiex = HJI_Example()
    #hjiex.plot()
    #hjiex.make_dataset()
    ssgex = SSG_Example()
    ssgex.make_ssdataset(isPlot=True)

