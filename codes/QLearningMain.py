# -*- coding: utf-8 -*-
import gym
import numpy as np
import pdb
import QLearning as ql

#------------------
# 1. 強化学習の環境の作成と変数の設定
# 割引報酬和の系列格納
sumReward = 0
sumRewards = []

# ε-greedyの初期化
epsilon = 0.5

# 状態の分割数
nSplit=50

# 反復回数
nIte = 50001
#------------------

#------------------
# 2. QLearningクラスのインスタンス化
agent = ql.QLearning(env='MountainCar-v0',gamma=0.99,nSplit=nSplit)
#------------------

#------------------
# 3. Q学習のエピソードのループ
for episode in np.arange(nIte):
    
    # 4. 環境の初期化
    x = agent.reset()

    # 5. ε-方策のεの値を減衰
    epsilon *= 0.999
        
    # 6. Q学習のステップのループ
    while(1):
        
        # 7. 行動を選択
        y = agent.selectAction(x,epsilon=epsilon)

        # 8. 行動を実行
        next_x,r,done = agent.doAction(y)

        # 9. Qテーブルの更新
        agent.update(x,y,next_x,r)

        # 10. 環境の描画
        if not episode%5000:
            agent.draw()
        
        # 11. 状態の更新
        x = next_x

        # 12. 報酬和の計算、初期化および記録
        if not done:
            sumReward += agent.gamma**agent.step * r
        else:
            sumRewards.append(sumReward)
            sumReward = 0
            break

    print(f"Episode:{episode},sum of rewards:{sumRewards[-1]}")

    # 13. 強化学習環境の終了
    agent.close()

    # 14. 割引報酬和とQ関数のプロット
    if episode==1000 or not episode%5000:
        agent.plotEval(sumRewards,fName=f"../results/Qlearning_sumRewards_{nSplit}_{episode}.pdf")
        agent.plotModel2D(xLabel="位置",yLabel="速度",fName=f"../results/Qlearning_Qtable_{nSplit}_{episode}.pdf")
#------------------