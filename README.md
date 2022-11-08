# mindspore-baselines

MindSpore version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3-688), for supporting reinforcement learning research

## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- MindSpore == 1.9.0
- 笔记本电脑
### Installation
1. **Clone repo**
    
    ```bash
    git clone https://github.com/superboySB/mindspore-baselines.git && cd mindspore-baselines
    ```
    
2. [Optional] Create Virtual Environment for GPU

   ```sh
   sudo apt-get install libgmp-dev
   wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
   sudo sh ./cuda_11.1.1_455.32.00_linux.run --override
   echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
   echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
   source ~/.bashrc
   
   # cudnn needs license
   wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz
   tar -zxvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
   sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
   sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
   
   # Install mindspore-gpu
   conda create -n msrl python==3.7
   conda activate msrl
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   
3. **Install minimal dependent packages**
    
    ```shell
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -e .[docs,tests,extra]
    ```
    
4. [Optional] All unit tests in mindspore-baselines3 can be run using `pytest` runner:

    ```
    make pytest
    ```

5. [Optional] If you want to install all of RL environments in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo), run:
    
    ```sh
    # mujuco
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    mkdir ~/.mujoco
    tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
    cd ~/.mujoco/mujoco210/bin && ./simulate
	pip install -U 'mujoco-py<2.2,>=2.1'
	
	# Others
	pip install -e .[develop]
	```

## :computer: Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gym
import mindspore as ms
from mindspore_baselines import PPO

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", device_id=0)

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```
from mindspore_baselines3 import PPO

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", device_id=0)
model = PPO("MlpPolicy", "CartPole-v1").learn(10000)
```

Please refer to the [documentation](https://stable-baselines3.readthedocs.io/) for more examples and use our repo as same as SB3 .


## :checkered_flag: Testing & Rendering
We will evaluate the trained model here.
```
\comming soon!
```

## :page_facing_up: Q&A
Q: Cannot render the results
> libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)

A: Modify the conda env

```sh
cd /home/$USER/miniconda3/envs/msrl/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29
```

## :clap: Reference
This codebase is based on SB3 and msrl which are open-sourced. Please refer to that repo for more documentation.
- Stable Baselines3 (https://github.com/DLR-RM/stable-baselines3)
- MindSpore/reinforcement (msrl) (https://gitee.com/mindspore/reinforcement)

## :e-mail: Contact
If you have any question, please email `604896160@qq.com`.

​	
