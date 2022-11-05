# mindspore-baselines

MindSpore version of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), for supporting reinforcement learning research

## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- MindSpore == 1.9.0
- 笔记本电脑
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/superboySB/mindspore-baselines.git && cd mindspore-baselines
    ```

2. [Optional] Create Virtual Environment for GPU

   ```sh
   # 需要GPU的话，可以先测试单机GPU版本的ms是否可用，若使用Ascend请参考官网。
   sudo apt-get install libgmp-dev
   wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
   sudo sh ./cuda_11.1.1_455.32.00_linux.run --override
   echo -e "export PATH=/usr/local/cuda-11.1/bin:\$PATH" >> ~/.bashrc
   echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
   source ~/.bashrc
   
   # 然后还得自己像早期TF一样搞cudnn...（以下仅为了mark版本，需要有license）
   wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz
   tar -zxvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
   sudo cp cuda/include/cudnn.h /usr/local/cuda-11.1/include
   sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
   sudo chmod a+r /usr/local/cuda-11.1/include/cudnn.h /usr/local/cuda-11.1/lib64/libcudnn*
   
   # 安装GPU版本的ms
   conda create -n msrl python==3.7
   conda activate msrl
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. Install minimal dependent packages
    ```sh
    # 安装CPU版本的ms:
    pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 安装主要支持
    pip install -e .[docs,tests,extra]
    ```

4. All unit tests in mindspore-baselines3 can be run using `pytest` runner:

    ```
    make pytest
    ```

5. [Optional] If you want to install all of RL environments in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo), run:
    ```sh
    # 安装mujuco
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    mkdir ~/.mujoco
    tar -zxvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
    cd ~/.mujoco/mujoco210/bin && ./simulate
    pip install -U 'mujoco-py<2.2,>=2.1'
	
	# 安装其他
	pip install -e .[develop]
	```

## :computer: Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gym
import mindspore as ms
from mindspore_baselines3 import PPO

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
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10000)
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more examples.



## :computer: Training

### DQN

```shell
python test/discrete/test_dqn.py
python examples/atari/atari_dqn.py --device GPU --task PongNoFrameskip-v4
# 可得到与torch类似或更快的收敛速度，但物理用时慢了3倍左右
```

### PG

```shell
python test/discrete/test_pg.py 
# 暂时遇到ops.multinomial算子不稳定的问题，影响了所有基于dist.sample()决策的算法性能，已提交issue
```

## :checkered_flag: Testing & Rendering
To evaluate the trained model, using the following command:
```
comming soon!
```

## :page_facing_up: Q&A
Q: Meet this issue when rendering
> libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)

A: modify the conda env

```sh
cd /home/$USER/miniconda3/envs/msrl/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.29
```

Q：亲测用conda配置cudatoolkit+cudnn的运行版本是报错的（但是人家pytorch就可以）

A：因为[源代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/run_check/_check_version.py)会自动扫/usr/local的version，且直接写死成了10.1和11.1。目前实际运行结果是，GPU与CPU训练所需要step相近，速度提升取决于设备性能。注意：mindrl所有测试脚本中均可通过设置device切换gpu/cpu/ascend，与 PyTorch 不同的是，一旦设备设置成功，输入数据和模型会默认拷贝到指定的设备中执行，不需要也无法再改变数据和模型所运行的设备类型，模型只有在正向传播阶段才会自动记录反向传播需要的梯度，而在推理阶段不会默认记录grad。

## :clap: Reference
This codebase is based on adept and Ray which are open-sourced. Please refer to that repo for more documentation.
- tianshou (https://github.com/thu-ml/tianshou)
- MindSpore/reinforcement (https://gitee.com/mindspore/reinforcement)

## :e-mail: Contact
If you have any question, please email `604896160@qq.com`.

​	
