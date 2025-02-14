import torch as th
from src.MLagent import MLAgent,MLAgent_Passification,MLAgent_HJI,MLAgent_SSG
from plots.plot_model import plot_pendulum,plot_cartpole,plot_vanderpol,plot_benchmark,plot_hjbex,plot_ssgex
from src.dynamics import Dynamics,CartPole,Pendulum,VanDerPol,BenchmarkExample,HJI_Example, SSG_Example
from src.models import Stable_Dynamics,Safty_Dynamics,L2_Dynamics,Passification_Dynamics,HJI_Dynamics, SSG_Dynamics, SSG_DynamicsFW

if __name__ == "__main__":
    """ Example 2.9 """
    # agent = MLAgent(name = "cartpole",
    #                 dynamics=CartPole,
    #                 model_class=Stable_Dynamics,
    #                 batch_size=10096,
    #                 data_size=51,
    #                 lr=0.01,
    #                 device="mps")
    # model = agent.train(10)
    # #model = agent.load_model()
    # plot_cartpole(name= "cartpole",model = model,device=agent.device)

    """ Example 2.10 """
    # agent = MLAgent(name = "stable_vanderpol",
    #                 dynamics=VanDerPol,
    #                 model_class=Stable_Dynamics,
    #                 batch_size=10240,
    #                 data_size=101,
    #                 lr=0.01,
    #                 device="mps")
    # model = agent.train(epoches=20)
    # #model = agent.load_model()
    # plot_vanderpol(name="stable_vanderpol", model=model,device=agent.device)

    # """ Example 3.2 """
    # def eta(x:th.Tensor)->th.Tensor: return  1 - ((x[:, 0] - 1.5)**2 + x[:, 1]**2)
    # agent = MLAgent(name = "safe_vanderpol",
    #                 dynamics=VanDerPol,
    #                 model_class=Safty_Dynamics,
    #                 batch_size=5120,
    #                 data_size=101,
    #                 lr=0.01,
    #                 model_kwargs={"c4":2,"eta":eta},
    #                 device="mps")
    # model = agent.train(epoches=40)
    # #model = agent.load_model()
    # plot_vanderpol(name = "safe_vanderpol", model = model,IsSafe=True, device= agent.device)

    # """ Example 3.4 """
    # agent = MLAgent(name = "L2_vanderpol",
    #                 dynamics=VanDerPol,
    #                 model_class=L2_Dynamics,
    #                 batch_size=5120,
    #                 data_size=101,
    #                 lr=0.01,
    #                 model_kwargs={"gamma":1},
    #                 device="mps")
    # #model = agent.train(20)
    # model = agent.load_model()
    # plot_vanderpol(name ="L2_vanderpol", model=model, IsL2=True,device=agent.device)

    """ Example 3.6 """
    # agent = MLAgent_Passification(name = "benchmark",
    #                 dynamics=BenchmarkExample,
    #                 model_class=Passification_Dynamics,
    #                 batch_size=5120,
    #                 data_size=41,
    #                 lr=0.01,
    #                 lambda_h= 1,
    #                 device="mps")
    # model = agent.train(40)
    # #model = agent.load_model()
    # plot_benchmark(name = "benchmark",model = model,device=agent.device)
    
    # """ Example 3.9 """
    # def q_fnc(x:th.Tensor)->th.Tensor: return x ** 2
    # def R_fnc(x:th.Tensor)->th.Tensor: return th.ones_like(x,dtype=th.float32,device=x.device) 
    # agent = MLAgent_HJI(name = "HJI_example",
    #                 dynamics=HJI_Example,
    #                 model_class=HJI_Dynamics,
    #                 model_kwargs={"q_fnc":q_fnc,"R_fnc":R_fnc,"net_arch":[128,32]},
    #                 batch_size=512,
    #                 data_size=256,
    #                 lr=0.01,
    #                 lambda_h=0.002,
    #                 device="mps")
    # model = agent.train(8000)
    # #model = agent.load_model()
    # plot_hjbex("HJI_example",model=model,device=agent.device)

    agent = MLAgent_SSG(name = "SSG_example",
                    dynamics=SSG_Example,
                    model_class=SSG_DynamicsFW,
                    batch_size=10120,
                    data_size=5,
                    model_kwargs={"net_arch":[128,32]},
                    lr=0.01,
                    lambda_h=0.01,
                    device="mps")
    model = agent.train(1000)
    #model = agent.load_model()
    plot_ssgex(name = "SSG_example",model=model,device=agent.device)



        
       
    