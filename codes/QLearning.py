# -*- coding: utf-8 -*-
import gym
import numpy as np
import pdb
import matplotlib.pylab as plt

class QLearning:
    #------------------
    # 1. 強化学習の環境および変数の初期化
    # env: 強化学習タスク環境名
    # gamma: 割引率（実数スカラー）
    # nSplit: 状態の分割数（整数スカラー）
    def __init__(self,env,gamma=0.99,nSplit=50):
    
        # 環境の読み込み
        self.env = gym.make(env)
        
        # 割引率
        self.gamma = gamma

        # 行動数
        self.nAction = self.env.action_space.n
        
        # 各状態の最小値と最大値
        self.stateMin = self.env.observation_space.low
        self.stateMax = self.env.observation_space.high

        # 状態の分割数
        self.nSplit = nSplit
        self.cellWidth = (self.stateMax-self.stateMin)/self.nSplit
        
        # Qテーブルの初期化
        self.Q = np.zeros((self.nSplit,self.nSplit,self.nAction))
    #------------------
    
    #------------------
    # 2. 状態および各種変数の初期化
    def reset(self):
        
        # 状態の初期化
        state = self.env.reset()
        
        # ステップの初期化
        self.step = 0
        
        return state
    #------------------
    
    #------------------
    # 3. 状態のインデックス取得
    # state：状態（実数ベクトル）
    def getStateIndex(self,state):
    
        # 離散値に変換
        stateInd = ((state-self.stateMin)/self.cellWidth).astype(int)

        return stateInd
    #------------------
    
    #------------------
    # 4. 行動の選択
    # state: 状態ベクトル
    def selectAction(self,state,epsilon=0.02):
        # 状態の離散化
        stateInd = self.getStateIndex(state)
        
        # ε-貪欲方策
        if np.random.uniform(0,1) > epsilon:
            action = np.argmax(self.Q[stateInd[0]][stateInd[1]])
        else:
            action = np.random.randint(self.nAction)

        return action
    #------------------
    
    #------------------
    # 5. 行動の実行、描画およびタスクの終了判定
    # action: 行動インデックス（整数スカラー）
    def doAction(self,action):

        # 行動の実行、次の状態・報酬・ゲーム終了FLG・詳細情報を取得
        next_state,reward,done,_ = self.env.step(action)

        # ステップを1増加
        self.step += 1

        return next_state,reward,done
    #------------------

    #------------------
    # 6. Qテーブルの更新
    # state：現在の状態（実数ベクトル）
    # action：行動インデックス（整数スカラー）
    # next_state：次の状態（実数ベクトル）
    # reward：報酬値（実数スカラー）
    # alpha：学習率（実数スカラー、デフォルトでは0.2）
    def update(self,state,action,next_state,reward,alpha=0.2):
        # 状態の離散化
        stateInd = self.getStateIndex(state)
        next_stateInd = self.getStateIndex(next_state)
        
        # 行動後の状態で得られる最大のQ値 Q(s',a')
        next_max_Qvalue = np.max(self.Q[next_stateInd[0]][next_stateInd[1]])

        # 行動前の状態のQ値 Q(s,a)
        Qvalue = self.Q[stateInd[0]][stateInd[1]][action]
        
        # Q関数の更新
        self.Q[stateInd[0]][stateInd[1]][action] = Qvalue + alpha * (reward+self.gamma*next_max_Qvalue-Qvalue)
    #------------------

    #------------------
    # 7. タスクの終了
    def close(self):
        self.env.close()
    #------------------
    
    #------------------
    # 8. 環境の描画
    def draw(self):
        self.env.render() 
    #------------------

    #------------------ 
    # 9. 評価のプロット
    # trEval：学習の評価
    # isSmoothing：平滑化のオンオフ（真偽値）
    # yLabel：y軸のラベル（文字列）
    # fName：画像の保存先（文字列）
    def plotEval(self,trEval,isSmoothing=True,yLabel="割引報酬和",fName=""):

        # 平滑化
        if isSmoothing:
            trEval = [np.mean(trEval[i-10:i]) for i in range(10,len(trEval))]
    
        # グラフのプロット
        plt.plot(trEval,'b',label="学習")
        
        # 各軸の範囲とラベルの設定
        plt.xlabel("エピソード")
        plt.ylabel(yLabel)

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
        plt.close()
    #------------------

    #------------------
    # 10. Qテーブルのプロット
    # xLabel: x軸のラベル（文字列）
    # yLabel: y軸のラベル（文字列）
    # fName：画像の保存先（文字列）
    def plotModel2D(self,xLabel="x1",yLabel="x2",fName="../figures/Qlearning_Qtable.png"):
    
        # 各軸の範囲設定
        X1 = np.arange(self.stateMin[0],self.stateMax[0],(self.stateMax[0]-self.stateMin[0])/self.nSplit)
        X2 = np.arange(self.stateMin[1],self.stateMax[1],(self.stateMax[1]-self.stateMin[1])/self.nSplit)
        
        # contourプロット
        CS = plt.contourf(X1,X2,np.max(self.Q,axis=2),linewidths=2,cmap="gray")

        # contourのカラーバー
        CB = plt.colorbar(CS)
        CB.ax.tick_params(labelsize=14)

        # 各軸の範囲とラベルの設定 
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
        plt.close()
    #------------------