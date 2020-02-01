# -*- coding: utf-8 -*-
import gym

#-------------------
# 1. MountainCarタスクの読み込み
env = gym.make('MountainCar-v0')
#-------------------

#-------------------
# 2. 環境の初期化
env.reset()
#-------------------

#-------------------
# 3. ランダムに行動を選択し描画
for i in range(500):
    env.step(env.action_space.sample())
    env.render()
#-------------------

#-------------------
# 4. 終了
env.close()
#-------------------