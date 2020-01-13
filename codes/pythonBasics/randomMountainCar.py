# -*- coding: utf-8 -*-
import gym

#-------------------
# 1. MountainCarタスクの読み込み
agent = gym.make('MountainCar-v0')
#-------------------

#-------------------
# 2. 環境の初期化
agent.reset()
#-------------------

#-------------------
# 3. ランダムに行動を選択し描画
for i in range(500):
    agent.step(agent.action_space.sample())
    agent.render()
#-------------------
#-------------------
# 4. 終了
agent.close()
#-------------------