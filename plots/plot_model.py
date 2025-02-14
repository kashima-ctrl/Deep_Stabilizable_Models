import torch as th
import numpy as np
import sys
sys.path.append("./")
import src.dynamics as dyn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dashed_circle(center_x=0, center_y=0, radius=1, color='blue', linestyle='--'):
    """
    Plots a dashed circle with specified center, radius, color, and line style.
    
    Parameters:
    - center_x (float): X-coordinate of the circle's center.
    - center_y (float): Y-coordinate of the circle's center.
    - radius (float): Radius of the circle.
    - color (str): Color of the circle's outline. Default is 'blue'.
    - linestyle (str): Line style for the circle's outline. Default is dashed ('--').
    """
    
    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, 100)  # Angle range for a full circle
    x = center_x + radius * np.cos(theta)   # X coordinates for the circle
    y = center_y + radius * np.sin(theta)   # Y coordinates for the circle

    # Plot the circle

    plt.plot(x, y, linestyle=linestyle, color=color, linewidth=4)  # Plot with specified color and style


def plot_cartpole(name, model:th.ScriptModule,device):
    plt.rc('text', usetex=True) # use latex
    label_size = 40
    font_size =44
    theta_max = np.pi/6
    theta_dot_max = 1.0
    theta = np.linspace(-theta_max, theta_max, 200)
    theta_dot = np.linspace(-theta_dot_max, theta_dot_max, 200)
    
    theta_tmp, theta_dot_tmp = np.meshgrid(theta,theta_dot)
    theta = theta_tmp.flatten()
    theta_dot = theta_dot_tmp.flatten()
    x, x_dot= np.zeros_like(theta), np.ones_like(theta)*0
    state = np.column_stack((x, x_dot, theta, theta_dot))
    state_tensor= th.tensor(state,dtype=th.float32,device=device)
    state_tensor.requires_grad = True

    f,g, alpha, _ = model(state_tensor)

    dx1 = (f[:,2,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f[:,3,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta_tmp,
                    theta_dot_tmp,
                    dx1.reshape(200,200),
                    dx2.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)
    
    plt.xlabel(r"$x_3$",fontsize=font_size)
    plt.ylabel(r"$x_4$",fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/cartpole_LDNA.png")
    print("LDNA...done.")

    f_close = f + g @ alpha
    dx1 = (f_close[:,2,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f_close[:,3,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta_tmp,theta_dot_tmp,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel(r"$x_3$",fontsize=font_size)
    plt.ylabel(r"$x_4$",fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/cartpole_LDLA.png")
    print("LDLA...done.")

    dynamics = dyn.CartPole()
    u= np.zeros_like(theta)
    state_dot = dynamics.state_dot(state=state,action=u)
    dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
    speed = np.sqrt(dtheta**2 + dtheta_dot**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta.reshape(200,200),
                    theta_dot.reshape(200,200),
                    dtheta.reshape(200,200),
                    dtheta_dot.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel(r"$x_3$",fontsize=font_size)
    plt.ylabel(r"$x_4$",fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/cartpole_RDNA.png")
    print("RDNA...done.")

    u= alpha.cpu().detach().numpy().flatten()
    state_dot = dynamics.state_dot(state=state,action=u)
    dtheta,dtheta_dot = state_dot[:,2],state_dot[:,3]
    speed = np.sqrt(dtheta**2 + dtheta_dot**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(theta.reshape(200,200),
                    theta_dot.reshape(200,200),
                    dtheta.reshape(200,200),
                    dtheta_dot.reshape(200,200),
                    color=speed.reshape(200,200),cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel(r"$x_3$",fontsize=font_size)
    plt.ylabel(r"$x_4$",fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/cartpole_RDLA.png")
    print("RDLA...done.")

def plot_vanderpol(name, model: th.ScriptModule, IsSafe=False, IsL2=False, IsShow=False ,device=None):
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    label_size = 30
    font_size = 40
    x1_max = 3.0
    x2_max = 3.0
    dynamics = dyn.VanDerPol()
    if IsL2:
        N,tau =   1000,0.04
        
        t_list = np.linspace(0,N*tau,N+1)
        x = np.array([[0.0,0.0]])
        x_open = np.array([[0.0,0.0]])
        x_list=[x]
        x_open_list = [x_open]
        x_tensor = th.tensor(x.reshape(1,2), dtype=th.float32,device=device,requires_grad=True)
        for i in range(N):
             f, g, alpha, V = model(x_tensor)
             action = alpha.reshape(1, 1).cpu().detach().numpy()
             x = x + tau * (dynamics.state_dot(state=x,action=action+np.sin([1.0000*i*tau])))
             x_open = x_open + tau * (dynamics.state_dot(state=x_open,action=np.sin([1.0000*i*tau])))
             x_list.append(x)
             x_open_list.append(x_open)
             x_tensor = th.tensor(x.reshape(1,2), dtype=th.float32,device=device,requires_grad=True)
        x_list = np.array(x_list)[:,:,0]
        x_open_list = np.array(x_open_list)[:,:,0]
        fig = plt.figure(figsize=(8, 6))
        plt.xlim(0,N*tau)
        plt.ylim(-3.2,3.2)
        plt.plot(t_list, x_list, label=r"closed-loop", linewidth=4, color="b",
                  linestyle='-', marker='o',markevery=100, markersize=10)
        plt.plot(t_list, x_open_list, label=r"open-loop", linewidth=4,color="r",
                  linestyle='--', marker='s',markevery=100,markersize=10)
        plt.plot(t_list, np.sin(1.0000*t_list), label=r"$\sin(t)$", linewidth=4,color="gray",
                 linestyle='-.', marker='^',markevery=100,markersize=10)
        plt.xlabel(r"$t$", fontsize=font_size-5)
        plt.ylabel(r"$z(t)$", fontsize=font_size-5)
        plt.legend(fontsize=font_size-10)
        plt.tick_params(labelsize=label_size)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./figures/{name}/vanderpol_RDLAL2.png")
        print("learned controller L2 (RDLAL2)...done.")

        x = np.array([[0.0,0.0],[0.0,0.0]])
        x_list=[x]
        x_tensor = th.tensor(x.reshape(2,2), dtype=th.float32,device=device,requires_grad=True)
        for i in range(N):
             f, g, alpha, V = model(x_tensor)
             x_dot = f.reshape(2, 2).cpu().detach().numpy()
             
             action = th.hstack((g[0]@(alpha[0]+np.sin(1.0000*i*tau)),
                                 g[1]*(np.sin(1.0000*i*tau)))).T
             x = x + tau * (x_dot+action.detach().cpu().numpy())
             x_list.append(x)
             x_tensor = th.tensor(x.reshape(2,2), dtype=th.float32,device=device,requires_grad=True)
        x_clist = np.array(x_list)[:,0,0]
        x_olist = np.array(x_list)[:,1,0]
        fig = plt.figure(figsize=(8, 6))
        plt.xlim(0,N*tau)
        plt.ylim(-3.2,3.2)
        plt.plot(t_list, x_clist, label=r"closed-loop", linewidth=4, color="b",
                  linestyle='-', marker='o',markevery=100, markersize=10)
        plt.plot(t_list, x_olist, label=r"open-loop", linewidth=4,color="r",
                  linestyle='--', marker='s',markevery=100,markersize=10)
        plt.plot(t_list, np.sin(1.0000*t_list), label=r"$\sin(t)$", linewidth=4,color="gray",
                 linestyle='-.', marker='^',markevery=100,markersize=10)
        plt.xlabel(r"$t$", fontsize=font_size-5)
        plt.ylabel(r"$z(t)$", fontsize=font_size-5)
        plt.legend(fontsize=font_size-10)
        plt.tick_params(labelsize=label_size)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./figures/{name}/vanderpol_LDLAL2.png")
        print("learned controller L2 (LDLAL2)...done.")
    # Set up state space for plotting
    x1 = np.linspace(-x1_max, x1_max, 200)
    x2 = np.linspace(-x2_max, x2_max, 200)
    x1_tmp, x2_tmp = np.meshgrid(x1, x2)
    x1 = x1_tmp.flatten()
    x2 = x2_tmp.flatten()
    state = np.column_stack((x1, x2))
    state_tensor = th.tensor(state, dtype=th.float32,device=device)
    state_tensor.requires_grad = True

    # Obtain the dynamics (f) and control term (g, alpha) from the model
    f, g, alpha, V = model(state_tensor)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x1_tmp, x2_tmp, alpha.reshape(200, 200).cpu().detach().numpy(), cmap='viridis')  

    ax.set_xlabel(r"$x_1$", fontsize=font_size, labelpad =20)
    ax.set_ylabel(r"$x_2$", fontsize=font_size, labelpad =20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\alpha(x)$', fontsize=font_size,rotation=90, labelpad =20)
    ax.tick_params(labelsize=label_size)
    ax.view_init(elev=18, azim=55)

    plt.savefig(f"./figures/{name}/vanderpol_LA.png")
    print("learned controller (LA)...done.")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x1_tmp, x2_tmp, V.reshape(200, 200).cpu().detach().numpy(), cmap='viridis',alpha=0.8)  
    surf0 = ax.plot_surface(x1_tmp, x2_tmp, np.zeros((200,200)), color="gray", label="0-plane",alpha = 0.8)
    ax.set_xlabel(r"$x_1$", fontsize=font_size, labelpad =20)
    ax.set_ylabel(r"$x_2$", fontsize=font_size, labelpad =20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$V(x)$', fontsize=font_size,rotation=90, labelpad =20)
    ax.view_init(elev=10, azim=-103) 
    ax.tick_params(labelsize=label_size)

    plt.savefig(f"./figures/{name}/vanderpol_LV.png")
    print("learned V (LV)...done.")

    # Plot the open-loop dynamics (without control)
    dx1 = f[:, 0].reshape(200, 200).cpu().detach().numpy()
    dx2 = f[:, 1].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1, dx2, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/vanderpol_LDNA.png")
    print("Open-loop dynamics (LDNA)...done.")

    # Plot the closed-loop dynamics (with control)
    f_close = f + g @ alpha
    dx1 = f_close[:, 0].reshape(200, 200).cpu().detach().numpy()
    dx2 = f_close[:, 1].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1, dx2, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/vanderpol_LDLA.png")
    print("Closed-loop dynamics (LDLA)...done.")



    # Plot real dynamics without control using `VanDerPol` class directly
   
    u = np.zeros_like(x1)
    state_dot = dynamics.state_dot(state=state, action=u)
    dx1_real, dx2_real = state_dot[:, 0], state_dot[:, 1]
    speed = np.sqrt(dx1_real**2 + dx2_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1_real.reshape(200, 200), dx2_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/vanderpol_RDNA.png")
    print("Real dynamics without control (RDNA)...done.")

    # Plot real dynamics with control
    u = alpha.cpu().detach().numpy().flatten()
    state_dot = dynamics.state_dot(state=state, action=u)
    dx1_real, dx2_real = state_dot[:, 0], state_dot[:, 1]
    speed = np.sqrt(dx1_real**2 + dx2_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1_real.reshape(200, 200), dx2_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    if IsSafe:
        plot_dashed_circle(1.5,0)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/vanderpol_RDLA.png")
    print("Real dynamics with control (RDLA)...done.")
    if IsShow:
        plt.show()

def plot_benchmark(name, model: th.ScriptModule ,device=None):
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    label_size = 30
    font_size = 40
    x1_max = 1.0
    x2_max = 1.0
    x3_max = 1.0
    dynamics = dyn.BenchmarkExample()
    fig = plt.figure(figsize=(8, 6))
    N,Nx = 80,10
    t_list = np.linspace(0,N*dynamics.tau,N+1)

    x = np.random.uniform([0.3,0.3,0.3],[0.6,0.6,0.6],size=(Nx,3))
    energy_list=[np.zeros((Nx))]
    V_list = [np.zeros((Nx))]
    x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    for i in range(N):
        f, g, alpha, V, h, beta = model(x_tensor)
        alpha = alpha.flatten().cpu().detach().numpy()
        beta = beta.flatten().cpu().detach().numpy()
        u_bar = +1
        u = alpha+beta*u_bar
        x = x + dynamics.tau * (dynamics.state_dot(state=x,action=u.reshape(Nx,1)))
        V_list.append(V.flatten().cpu().detach().numpy())
        energy_list.append(energy_list[i]+dynamics.tau*x[:,0]*u_bar)
        x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    V_list = np.array(V_list)-V_list[1]
    V_list[0] = np.zeros(Nx)
    energy_list = np.array(energy_list)
    
    #plt.ylim(-1.2,1.2)
    plt.plot(t_list, V_list-energy_list, linewidth=2, color="blue",
                linestyle='-.',alpha = 0.4)
    
    x = np.random.uniform([0.3,0.3,0.3],[0.6,0.6,0.6],size=(Nx,3))
    energy_list=[np.zeros((Nx))]
    V_list = [np.zeros((Nx))]
    x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    for i in range(N):
        f, g, alpha, V, h, beta = model(x_tensor)
        alpha = alpha.flatten().cpu().detach().numpy()
        beta = beta.flatten().cpu().detach().numpy()
        u_bar = -1
        u = alpha+beta*u_bar
        x = x + dynamics.tau * (dynamics.state_dot(state=x,action=u.reshape(Nx,1)))
        V_list.append(V.flatten().cpu().detach().numpy())
        energy_list.append(energy_list[i]+dynamics.tau*x[:,0]*u_bar)
        x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    V_list = np.array(V_list)-V_list[1]
    V_list[0] = np.zeros(Nx)
    energy_list = np.array(energy_list)
    
    #plt.ylim(-1.2,1.2)
    plt.plot(t_list, V_list-energy_list, linewidth=2, color="red",
                linestyle='--',alpha = 0.4)
    
    x = np.random.uniform([0.3,0.3,0.3],[0.6,0.6,0.6],size=(Nx,3))
    energy_list=[np.zeros((Nx))]
    V_list = [np.zeros((Nx))]
    x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    for i in range(N):
        f, g, alpha, V, h, beta = model(x_tensor)
        alpha = alpha.flatten().cpu().detach().numpy()
        beta = beta.flatten().cpu().detach().numpy()
        u_bar = 2*np.sin(100*dynamics.tau*i)
        
        u = alpha+beta*u_bar
        x = x + dynamics.tau * (dynamics.state_dot(state=x,action=u.reshape(Nx,1)))
        V_list.append(V.flatten().cpu().detach().numpy())
        energy_list.append(energy_list[i]+dynamics.tau*x[:,0]*u_bar)
        x_tensor = th.tensor(x.reshape(Nx,3), dtype=th.float32,device=device,requires_grad=True)
    V_list = np.array(V_list)-V_list[1]
    V_list[0] = np.zeros(Nx)
    energy_list = np.array(energy_list)
    
    #plt.ylim(-1.2,1.2)
    plt.plot(t_list, V_list-energy_list, linewidth=2, color="gray",
                linestyle='-',alpha = 0.6)
    
    plt.xlim(0,N*dynamics.tau)
    plt.xlabel(r"$t$", fontsize=font_size-10)
    plt.ylabel(r"$V(x(t)) - V(x(0)) - \int_{0}^{t} {\bar u}^\top(\tau) y(\tau) d\tau$", fontsize=font_size-20)
    plt.tick_params(labelsize=label_size-5)
    plt.grid()
    
    custom_legend_with_markers = [
    plt.Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label=r"$\bar u = 2\sin(10t)$"),
    plt.Line2D([0], [0], color='blue', linestyle='-.',  linewidth=2, label=r"$\bar u =1$"),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=r"$\bar u =-1$")
    ]
    plt.legend(handles=custom_legend_with_markers,
               fontsize=font_size-15)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/banchmark_RDLApass.png")
    print("learned controller L2 (RDLAL2)...done.")
    #Set up state space for plotting
    x2 = np.linspace(-x2_max, x2_max, 200)
    x3 = np.linspace(-x3_max, x3_max, 200)
    x2_tmp, x3_tmp = np.meshgrid(x2, x3)
    x2 = x2_tmp.flatten()
    x3 = x3_tmp.flatten()
    x1 = np.ones_like(x2)*1

    state = np.column_stack((x1, x2, x3))
    state_tensor = th.tensor(state, dtype=th.float32,device=device)
    state_tensor.requires_grad = True

    # Obtain the dynamics (f) and control term (g, alpha) from the model
    f, g, alpha, V, h, beta = model(state_tensor)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x2_tmp, x3_tmp, alpha.reshape(200, 200).cpu().detach().numpy(), cmap='viridis')  

    ax.set_xlabel(r"$x_2$", fontsize=font_size, labelpad =20)
    ax.set_ylabel(r"$x_3$", fontsize=font_size, labelpad =20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\alpha(x)$', fontsize=font_size,rotation=90, labelpad =20)
    ax.tick_params(labelsize=label_size)
    ax.view_init(elev=18, azim=55)

    plt.savefig(f"./figures/{name}/benchmark_LA.png")
    print("learned controller (LA)...done.")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x2_tmp, x3_tmp, V.reshape(200, 200).cpu().detach().numpy(), cmap='viridis',alpha=0.8)  
    surf0 = ax.plot_surface(x2_tmp, x3_tmp, np.zeros((200,200)), color="gray", label="0-plane")
    ax.set_xlabel(r"$x_2$", fontsize=font_size, labelpad =20)
    ax.set_ylabel(r"$x_3$", fontsize=font_size, labelpad =20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$V(x)$', fontsize=font_size,rotation=90, labelpad =20)
    ax.view_init(elev=10, azim=-103) 
    ax.tick_params(labelsize=label_size)

    plt.savefig(f"./figures/{name}/benchmark_LV.png")
    print("learned V (LV)...done.")

    # Plot the open-loop dynamics (without control)
    dx2 = f[:, 1].reshape(200, 200).cpu().detach().numpy()
    dx3 = f[:, 2].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx2**2 + dx3**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x2_tmp, x3_tmp, dx2, dx3, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_2$", fontsize=font_size)
    plt.ylabel(r"$x_3$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/benchmark_LDNA.png")
    print("Open-loop dynamics (LDNA)...done.")

    # Plot the closed-loop dynamics (with control)
    f_close = f + g @ f[:,0,:].unsqueeze(2)
    dx2 = f_close[:, 1].reshape(200, 200).cpu().detach().numpy()
    dx3 = f_close[:, 2].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx2**2 + dx3**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x2_tmp, x3_tmp, dx2, dx3, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_2$", fontsize=font_size)
    plt.ylabel(r"$x_3$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/benchmark_LDLA.png")
    print("Closed-loop dynamics (LDLA)...done.")



    # Plot real dynamics without control using `benchmark` class directly
   
    u = np.zeros_like(x2)
    state_dot = dynamics.state_dot(state=state, action=u)
    dx2_real, dx3_real = state_dot[:, 1], state_dot[:, 2]
    speed = np.sqrt(dx2_real**2 + dx3_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x2_tmp, x3_tmp, dx2_real.reshape(200, 200), dx3_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_2$", fontsize=font_size)
    plt.ylabel(r"$x_3$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/benchmark_RDNA.png")
    print("Real dynamics without control (RDNA)...done.")

    # Plot real dynamics with control
    u = f[:,0,:].cpu().detach().numpy().flatten()
    state_dot = dynamics.state_dot(state=state, action=u)
    dx2_real, dx3_real = state_dot[:, 1], state_dot[:, 2]
    speed = np.sqrt(dx2_real**2 + dx3_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x2_tmp, x3_tmp, dx2_real.reshape(200, 200), dx3_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    
    plt.xlabel(r"$x_2$", fontsize=font_size)
    plt.ylabel(r"$x_3$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/benchmark_RDLA.png")
    print("Real dynamics with control (RDLA)...done.")

def plot_pendulum(model:th.ScriptModule):
    x1 = np.linspace(-3, 3, 200)
    x2 = np.linspace(-3, 3, 200)
    
    x, y = np.meshgrid(x1,x2)
    state = th.tensor(np.vstack((x.flatten(),y.flatten())).T,
                        dtype=th.float32)
    state.requires_grad = True
    f, g, alpha, _ = model(state)
 
    dx1 = (f[:,0,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f[:,1,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel("$x_1$",fontsize=18)
    plt.ylabel("$x_2$",fontsize=18)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig("./figures/pendulum/learned_dynamics.png")

    f_close = f + g @ alpha
    dx1 = (f_close[:,0,0].reshape(200,200)).cpu().detach().numpy()
    dx2 = (f_close[:,1,0].reshape(200,200)).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)
    
    plt.figure(figsize=(8,8))
    plt.streamplot(x,y,dx1,dx2,color=speed,cmap='autumn',arrowsize=5,linewidth=3.0, zorder=1)

    plt.xlabel("$x_1$",fontsize=18)
    plt.ylabel("$x_2$",fontsize=18)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig("./figures/pendulum/learned_fb_dynamics.png")

def plot_hjbex(name, model: th.ScriptModule ,device=None):
    #Set up state space for plotting
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    label_size = 30
    font_size = 40
    state = np.linspace(-3, 3, 200).reshape(200,1)
    
    state_tensor = th.tensor(state, dtype=th.float32,device=device)
    state_tensor.requires_grad = True

    # Obtain the dynamics (f) and control term (g, alpha) from the model
    f, g, alpha, V, H = model(state_tensor)
    # Plot the open-loop dynamics (without control)
    state_dot = f[:, 0].cpu().detach().numpy()

    plt.figure(figsize=(8, 8))
    plt.plot(state, state_dot, label=r"$f(x)$")
    plt.plot(state, state**2, label =r"$f_{\rm true}(x)$")

    plt.xlabel(r"$x$", fontsize=font_size)
    plt.xlim([-3,3])
    plt.ylim([-1,10])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/hjbex_RDLDNA.png")
    print("Open-loop dynamics (RDLDNA)...done.")

    alpha = alpha[:, 0].cpu().detach().numpy()
    plt.figure(figsize=(8, 8))

    plt.plot(state, alpha, label=r"$\alpha(x)$")
    plt.plot(state, -state*(state+np.sqrt(2+state**2)),label=r"$\alpha_{\rm opt}(x)$")
    plt.xlabel(r"$x$", fontsize=font_size)
    plt.xlim([-3,3])
    plt.ylim([-10,3])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/hjbex_LA.png")
    print("Open-loop dynamics (LA)...done.")

    g = g[:, 0].cpu().detach().numpy()
    plt.figure(figsize=(8, 8))

    plt.plot(state, g, label=r"$g(x)$")
    plt.plot(state, np.ones_like(state),label=r"$g_{\rm true}(x)$")
    
    plt.xlabel(r"$x$", fontsize=font_size)
    plt.xlim([-3,3])
    plt.ylim([-10,3])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/hjbex_Lg.png")
    print("Open-loop dynamics (Lg)...done.")

    V = V[:, 0].cpu().detach().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.plot(state, V,label=r"$V(x)$")
    plt.plot(state, 
             (state**3 + (state**2+2)**(1.5) - 2**(1.5))/3,
             label=r"$V_{\rm true}(x)$")
    plt.xlabel(r"$x$", fontsize=font_size)

    plt.xlim([-3,3])
    plt.ylim([-1,25])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/hjbex_RVLV.png")
    print("Open-loop dynamics (LVNV)...done.")

    plt.figure(figsize=(8, 8))


    plt.plot(state, -alpha/g ,label=r"$\nabla V(x)$")
    plt.plot(state, 
             (state**2 + state*(state**2+2)**(0.5)),
             label=r"$\nabla V_{\rm true}(x)$")
    plt.xlabel(r"$x$", fontsize=font_size)
    plt.xlim([-3,3])
    plt.ylim([-5,20])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/hjbex_RVLVx.png")
    print("Open-loop dynamics (LVNV)...done.")

def plot_ssgex(name, model: th.ScriptModule ,device=None):
    from scipy.optimize import fsolve
    x1_max = 3.0
    x2_max = 3.0
    dynamics = dyn.SSG_Example()
    
    # Define the equation for x2
    def equation(x2, u): return 125 * x2**3 + x2 - u

    # Define the range of u values
    u_values = np.linspace(-3, 3, 200)  # Example: u values from -2 to 2
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


    #Set up state space for plotting
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    label_size = 30
    font_size = 40
    u = np.linspace(-3, 3, 200).reshape(200,1)
    u_tensor = th.tensor(u, dtype=th.float32,device=device)
    stable_state = model.h(u_tensor).cpu().detach().numpy()
    plt.figure(figsize=(8, 8))
    plt.plot(u, stable_state,"--",label=[r"$h_1(u)$",r"$h_2(u)$"])
    plt.plot(u_values, x1_solutions, label=r"$x_1 = 5x_2$", color='blue',alpha=0.5)
    plt.plot(u_values, x2_solutions, label=r"$x_2 $", color='orange',alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)


    plt.xlabel(r"$u$", fontsize=font_size)
    plt.xlim([-3,3])
    #plt.ylim([-1,10])
    plt.grid()
    plt.legend(fontsize = font_size -20)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/ssgex_h.png")
    print("h...done.")

    # Set up state space for plotting
    x1 = np.linspace(-x1_max, x1_max, 200)
    x2 = np.linspace(-x2_max, x2_max, 200)
    x1_tmp, x2_tmp = np.meshgrid(x1, x2)
    x1 = x1_tmp.flatten()
    x2 = x2_tmp.flatten()
    state = np.column_stack((x1, x2))
    state_tensor = th.tensor(state, dtype=th.float32,device=device)
    state_tensor.requires_grad = True
    u = np.ones((200**2,1))*0
    u_tensor = th.tensor(u,dtype=th.float32,device=device)

    # Obtain the dynamics (f) and control term (g, alpha) from the model
    f, h, V = model(state_tensor, u_tensor)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x1_tmp, x2_tmp, V.reshape(200, 200).cpu().detach().numpy(), cmap='viridis',alpha=0.8)  
    surf0 = ax.plot_surface(x1_tmp, x2_tmp, np.zeros((200,200)), color="gray", label="0-plane",alpha = 0.8)
    ax.set_xlabel(r"$x_1$", fontsize=font_size, labelpad =20)
    ax.set_ylabel(r"$x_2$", fontsize=font_size, labelpad =20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$V(x)$', fontsize=font_size,rotation=90, labelpad =20)
    ax.view_init(elev=10, azim=-103) 
    ax.tick_params(labelsize=label_size)

    plt.savefig(f"./figures/{name}/ssg_LV.png")
    print("learned V (LV)...done.")

    # Plot the open-loop dynamics (without control)
    dx1 = f[:, 0].reshape(200, 200).cpu().detach().numpy()
    dx2 = f[:, 1].reshape(200, 200).cpu().detach().numpy()
    speed = np.sqrt(dx1**2 + dx2**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1, dx2, color=speed, cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/ssg_LDNA.png")
    print("Open-loop dynamics (LDNA)...done.")

    # Plot real dynamics without control using `VanDerPol` class directly

    state_dot = dynamics.state_dot(state=state, action=u)
    dx1_real, dx2_real = state_dot[:, 0], state_dot[:, 1]
    speed = np.sqrt(dx1_real**2 + dx2_real**2)

    plt.figure(figsize=(8, 8))
    plt.streamplot(x1_tmp, x2_tmp, dx1_real.reshape(200, 200), dx2_real.reshape(200, 200), color=speed.reshape(200, 200),
                   cmap='autumn', arrowsize=5, linewidth=3.0, zorder=1)
    plt.xlabel(r"$x_1$", fontsize=font_size)
    plt.ylabel(r"$x_2$", fontsize=font_size)
    plt.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(f"./figures/{name}/ssg_RDNA.png")
    print("Real dynamics without control (RDNA)...done.")

    fnn =f.squeeze(2).detach().cpu().numpy()
    print("FIT_f=",100*(1-np.sqrt(((state_dot-fnn)**2).sum())/np.sqrt(((state_dot - state_dot.mean(0))**2).sum())))


