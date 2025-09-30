"""
贝尔曼期望方程与蒙特卡洛方法 (Bellman Expectation Equation & Monte Carlo Methods) 详细实现

本文件提供了贝尔曼期望方程和蒙特卡洛方法的完整实现，包括：
1. 贝尔曼期望方程的数学推导和实现
2. 蒙特卡洛预测方法（First-Visit MC, Every-Visit MC）
3. 蒙特卡洛控制方法（MC Control with ε-greedy）
4. 重要性采样方法（Importance Sampling）
5. 丰富的可视化功能和教学示例

作者：强化学习教学代码
版本：2.0
用途：深入理解贝尔曼期望方程和蒙特卡洛方法的数学原理和实际应用
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings
from collections import defaultdict, deque
import random
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BellmanExpectationEquation:
    """
    贝尔曼期望方程详细实现类
    
    贝尔曼期望方程是强化学习的核心数学基础：
    
    状态价值函数的贝尔曼期望方程：
    V^π(s) = E_π[R_{t+1} + γ*V^π(S_{t+1}) | S_t = s]
           = Σ_a π(a|s) * Σ_{s',r} p(s',r|s,a) * [r + γ*V^π(s')]
    
    动作价值函数的贝尔曼期望方程：
    Q^π(s,a) = E_π[R_{t+1} + γ*Q^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
             = Σ_{s',r} p(s',r|s,a) * [r + γ * Σ_{a'} π(a'|s') * Q^π(s',a')]
    
    核心概念：
    1. 期望：对所有可能的未来轨迹求期望
    2. 递归性：当前价值依赖于未来价值
    3. 策略依赖：价值函数依赖于所遵循的策略
    4. 马尔可夫性：未来只依赖于当前状态
    """
    
    def __init__(self, states: List[str], actions: List[str], 
                 transition_probs: Dict[Tuple[str, str, str], float],
                 rewards: Dict[Tuple[str, str, str], float], 
                 gamma: float = 0.9):
        """
        初始化贝尔曼期望方程求解器
        
        参数:
            states: 状态空间
            actions: 动作空间
            transition_probs: 转移概率 {(s, a, s'): P(s'|s,a)}
            rewards: 奖励函数 {(s, a, s'): R(s,a,s')}
            gamma: 折扣因子
        """
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.gamma = gamma
        
        # 构建索引映射
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        
        # 构建转移概率和奖励矩阵
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for (s, a, s_next), prob in transition_probs.items():
            i, j, k = self.state_to_idx[s], self.action_to_idx[a], self.state_to_idx[s_next]
            self.P[i, j, k] = prob
        
        for (s, a, s_next), reward in rewards.items():
            i, j, k = self.state_to_idx[s], self.action_to_idx[a], self.state_to_idx[s_next]
            self.R[i, j, k] = reward
        
        print(f"✅ 贝尔曼期望方程求解器初始化完成")
        print(f"   状态数量: {self.n_states}")
        print(f"   动作数量: {self.n_actions}")
        print(f"   折扣因子: {self.gamma}")
    
    def solve_bellman_expectation_v(self, policy: np.ndarray, 
                                   method: str = 'iterative',
                                   max_iterations: int = 1000,
                                   tolerance: float = 1e-8) -> np.ndarray:
        """
        求解状态价值函数的贝尔曼期望方程
        
        V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*V^π(s')]
        
        参数:
            policy: 策略矩阵 π(a|s)
            method: 求解方法 ('iterative' 或 'analytical')
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        返回:
            状态价值函数 V^π
        """
        print(f"\n🧮 求解状态价值函数贝尔曼期望方程 - {method}方法")
        print("=" * 60)
        
        if method == 'analytical':
            return self._solve_v_analytical(policy)
        else:
            return self._solve_v_iterative(policy, max_iterations, tolerance)
    
    def _solve_v_analytical(self, policy: np.ndarray) -> np.ndarray:
        """解析求解状态价值函数"""
        print("使用解析方法求解...")
        
        # 构建系数矩阵 A 和常数向量 b
        # (I - γ * P^π) * V = R^π
        A = np.eye(self.n_states)
        b = np.zeros(self.n_states)
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                pi_sa = policy[s, a]
                for s_next in range(self.n_states):
                    p_sas = self.P[s, a, s_next]
                    r_sas = self.R[s, a, s_next]
                    
                    # 构建系数矩阵
                    A[s, s_next] -= self.gamma * pi_sa * p_sas
                    
                    # 构建常数向量
                    b[s] += pi_sa * p_sas * r_sas
        
        print(f"系数矩阵 A 的条件数: {np.linalg.cond(A):.2e}")
        
        try:
            V = np.linalg.solve(A, b)
            print("✅ 解析求解成功")
            return V
        except np.linalg.LinAlgError:
            print("❌ 解析求解失败，矩阵奇异")
            return self._solve_v_iterative(policy, 1000, 1e-8)
    
    def _solve_v_iterative(self, policy: np.ndarray, max_iterations: int, tolerance: float) -> np.ndarray:
        """迭代求解状态价值函数"""
        print("使用迭代方法求解...")
        
        V = np.zeros(self.n_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.n_states):
                v_new = 0
                for a in range(self.n_actions):
                    pi_sa = policy[s, a]
                    for s_next in range(self.n_states):
                        p_sas = self.P[s, a, s_next]
                        r_sas = self.R[s, a, s_next]
                        v_new += pi_sa * p_sas * (r_sas + self.gamma * V_old[s_next])
                V[s] = v_new
            
            delta = np.max(np.abs(V - V_old))
            
            if iteration < 10 or iteration % 100 == 0:
                print(f"   迭代 {iteration:3d}: δ = {delta:.8f}")
            
            if delta < tolerance:
                print(f"✅ 在第 {iteration} 次迭代后收敛")
                break
        
        return V
    
    def solve_bellman_expectation_q(self, policy: np.ndarray, 
                                   max_iterations: int = 1000,
                                   tolerance: float = 1e-8) -> np.ndarray:
        """
        求解动作价值函数的贝尔曼期望方程
        
        Q^π(s,a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * Σ_{a'} π(a'|s') * Q^π(s',a')]
        
        参数:
            policy: 策略矩阵
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        返回:
            动作价值函数 Q^π
        """
        print(f"\n🎯 求解动作价值函数贝尔曼期望方程")
        print("=" * 60)
        
        Q = np.zeros((self.n_states, self.n_actions))
        
        for iteration in range(max_iterations):
            Q_old = Q.copy()
            
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    q_new = 0
                    for s_next in range(self.n_states):
                        p_sas = self.P[s, a, s_next]
                        r_sas = self.R[s, a, s_next]
                        
                        # 计算 Σ_{a'} π(a'|s') * Q^π(s',a')
                        expected_q = np.sum(policy[s_next] * Q_old[s_next])
                        
                        q_new += p_sas * (r_sas + self.gamma * expected_q)
                    
                    Q[s, a] = q_new
            
            delta = np.max(np.abs(Q - Q_old))
            
            if iteration < 10 or iteration % 100 == 0:
                print(f"   迭代 {iteration:3d}: δ = {delta:.8f}")
            
            if delta < tolerance:
                print(f"✅ 在第 {iteration} 次迭代后收敛")
                break
        
        return Q


class MonteCarloMethods:
    """
    蒙特卡洛方法详细实现类
    
    蒙特卡洛方法是一类基于采样的强化学习算法：
    
    核心思想：
    1. 通过与环境交互生成完整的回合序列
    2. 使用实际回报来估计价值函数
    3. 不需要环境的完整模型
    4. 适用于回合性任务
    
    主要方法：
    1. First-Visit MC: 只使用状态在回合中的首次访问
    2. Every-Visit MC: 使用状态在回合中的每次访问
    3. MC Control: 结合策略改进的蒙特卡洛控制
    4. 重要性采样: 处理off-policy学习
    """
    
    def __init__(self, states: List[str], actions: List[str], gamma: float = 0.9):
        """
        初始化蒙特卡洛方法
        
        参数:
            states: 状态空间
            actions: 动作空间
            gamma: 折扣因子
        """
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.gamma = gamma
        
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        
        # 存储学习历史
        self.learning_history = []
        self.episode_returns = []
        
        print(f"✅ 蒙特卡洛方法初始化完成")
        print(f"   状态数量: {self.n_states}")
        print(f"   动作数量: {self.n_actions}")
        print(f"   折扣因子: {self.gamma}")
    
    def first_visit_mc_prediction(self, episodes: List[List[Tuple]], 
                                 verbose: bool = True) -> np.ndarray:
        """
        First-Visit 蒙特卡洛预测
        
        对于每个回合中状态的首次访问，使用该访问之后的回报来更新价值估计
        
        参数:
            episodes: 回合列表，每个回合是 [(state, action, reward), ...] 的序列
            verbose: 是否显示详细信息
            
        返回:
            状态价值函数估计
        """
        if verbose:
            print(f"\n🎲 First-Visit 蒙特卡洛预测")
            print("=" * 50)
            print(f"   回合数量: {len(episodes)}")
        
        # 初始化
        returns = defaultdict(list)  # 存储每个状态的回报列表
        V = np.zeros(self.n_states)
        
        for episode_idx, episode in enumerate(episodes):
            # 计算每个时间步的回报
            G = 0
            episode_returns = []
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                episode_returns.append(G)
            episode_returns.reverse()
            
            # First-Visit: 只考虑状态的首次访问
            visited_states = set()
            for t, (state, action, reward) in enumerate(episode):
                if state not in visited_states:
                    visited_states.add(state)
                    returns[state].append(episode_returns[t])
            
            # 更新价值函数估计
            if episode_idx % 100 == 0 or episode_idx < 10:
                for state in self.states:
                    if state in returns:
                        V[self.state_to_idx[state]] = np.mean(returns[state])
                
                if verbose and (episode_idx % 100 == 0):
                    print(f"   回合 {episode_idx:4d}: 已访问状态数 = {len(returns)}")
        
        # 最终更新
        for state in self.states:
            if state in returns:
                V[self.state_to_idx[state]] = np.mean(returns[state])
        
        if verbose:
            print(f"✅ First-Visit MC预测完成")
            print(f"   访问过的状态数: {len(returns)}")
            for state in self.states:
                if state in returns:
                    visits = len(returns[state])
                    value = V[self.state_to_idx[state]]
                    print(f"   V({state}) = {value:.4f} (基于 {visits} 次访问)")
        
        return V
    
    def every_visit_mc_prediction(self, episodes: List[List[Tuple]], 
                                 verbose: bool = True) -> np.ndarray:
        """
        Every-Visit 蒙特卡洛预测
        
        对于每个回合中状态的每次访问，都使用该访问之后的回报来更新价值估计
        
        参数:
            episodes: 回合列表
            verbose: 是否显示详细信息
            
        返回:
            状态价值函数估计
        """
        if verbose:
            print(f"\n🎯 Every-Visit 蒙特卡洛预测")
            print("=" * 50)
            print(f"   回合数量: {len(episodes)}")
        
        # 初始化
        returns = defaultdict(list)
        V = np.zeros(self.n_states)
        
        for episode_idx, episode in enumerate(episodes):
            # 计算每个时间步的回报
            G = 0
            episode_returns = []
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                episode_returns.append(G)
            episode_returns.reverse()
            
            # Every-Visit: 考虑状态的每次访问
            for t, (state, action, reward) in enumerate(episode):
                returns[state].append(episode_returns[t])
            
            # 更新价值函数估计
            if episode_idx % 100 == 0 or episode_idx < 10:
                for state in self.states:
                    if state in returns:
                        V[self.state_to_idx[state]] = np.mean(returns[state])
                
                if verbose and (episode_idx % 100 == 0):
                    total_visits = sum(len(returns[s]) for s in returns)
                    print(f"   回合 {episode_idx:4d}: 总访问次数 = {total_visits}")
        
        # 最终更新
        for state in self.states:
            if state in returns:
                V[self.state_to_idx[state]] = np.mean(returns[state])
        
        if verbose:
            print(f"✅ Every-Visit MC预测完成")
            total_visits = sum(len(returns[s]) for s in returns if s in returns)
            print(f"   总访问次数: {total_visits}")
            for state in self.states:
                if state in returns:
                    visits = len(returns[state])
                    value = V[self.state_to_idx[state]]
                    print(f"   V({state}) = {value:.4f} (基于 {visits} 次访问)")
        
        return V
    
    def mc_control_epsilon_greedy(self, environment_simulator: Callable,
                                 num_episodes: int = 1000,
                                 epsilon: float = 0.1,
                                 epsilon_decay: float = 0.995,
                                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        蒙特卡洛控制 with ε-贪婪策略
        
        交替进行策略评估和策略改进：
        1. 使用当前策略生成回合
        2. 使用MC方法更新Q函数
        3. 使用ε-贪婪策略改进策略
        
        参数:
            environment_simulator: 环境模拟器函数
            num_episodes: 回合数
            epsilon: ε-贪婪参数
            epsilon_decay: ε衰减率
            verbose: 是否显示详细信息
            
        返回:
            (最优策略, Q函数)
        """
        if verbose:
            print(f"\n🎮 蒙特卡洛控制 (ε-贪婪策略)")
            print("=" * 50)
            print(f"   回合数: {num_episodes}")
            print(f"   初始ε: {epsilon}")
            print(f"   ε衰减率: {epsilon_decay}")
        
        # 初始化
        Q = np.zeros((self.n_states, self.n_actions))
        returns = defaultdict(list)
        policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
        episode_rewards = []
        epsilon_history = []
        
        for episode in range(num_episodes):
            # 生成回合
            episode_data = environment_simulator(policy, epsilon)
            states, actions, rewards = episode_data
            
            # 计算回报
            G = 0
            episode_returns = []
            for t in reversed(range(len(rewards))):
                G = rewards[t] + self.gamma * G
                episode_returns.append(G)
            episode_returns.reverse()
            
            # 更新Q函数 (First-Visit)
            visited_sa = set()
            for t in range(len(states)-1):  # 最后一个状态没有动作
                s, a = states[t], actions[t]
                sa_pair = (s, a)
                
                if sa_pair not in visited_sa:
                    visited_sa.add(sa_pair)
                    s_idx = self.state_to_idx[s]
                    a_idx = self.action_to_idx[a]
                    returns[sa_pair].append(episode_returns[t])
                    Q[s_idx, a_idx] = np.mean(returns[sa_pair])
            
            # 策略改进 (ε-贪婪)
            for s_idx in range(self.n_states):
                best_action = np.argmax(Q[s_idx])
                for a_idx in range(self.n_actions):
                    if a_idx == best_action:
                        policy[s_idx, a_idx] = 1 - epsilon + epsilon / self.n_actions
                    else:
                        policy[s_idx, a_idx] = epsilon / self.n_actions
            
            # 记录统计信息
            episode_reward = sum(rewards)
            episode_rewards.append(episode_reward)
            epsilon_history.append(epsilon)
            
            # ε衰减
            epsilon = max(0.01, epsilon * epsilon_decay)
            
            if verbose and (episode % 100 == 0):
                avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
                print(f"   回合 {episode:4d}: 平均奖励 = {avg_reward:6.2f}, ε = {epsilon:.4f}")
        
        self.episode_returns = episode_rewards
        self.learning_history = epsilon_history
        
        if verbose:
            print(f"✅ 蒙特卡洛控制完成")
            print(f"   最终ε: {epsilon:.4f}")
            print(f"   最后100回合平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
        
        return policy, Q
    
    def importance_sampling_prediction(self, episodes: List[List[Tuple]], 
                                     behavior_policy: np.ndarray,
                                     target_policy: np.ndarray,
                                     method: str = 'weighted',
                                     verbose: bool = True) -> np.ndarray:
        """
        重要性采样蒙特卡洛预测
        
        使用行为策略生成的数据来评估目标策略的价值函数
        
        参数:
            episodes: 回合列表 [(state, action, reward), ...]
            behavior_policy: 行为策略 (生成数据的策略)
            target_policy: 目标策略 (要评估的策略)
            method: 'ordinary' 或 'weighted' 重要性采样
            verbose: 是否显示详细信息
            
        返回:
            目标策略的状态价值函数估计
        """
        if verbose:
            print(f"\n🎭 重要性采样蒙特卡洛预测 ({method})")
            print("=" * 50)
            print(f"   回合数量: {len(episodes)}")
        
        if method == 'ordinary':
            return self._ordinary_importance_sampling(episodes, behavior_policy, target_policy, verbose)
        else:
            return self._weighted_importance_sampling(episodes, behavior_policy, target_policy, verbose)
    
    def _ordinary_importance_sampling(self, episodes, behavior_policy, target_policy, verbose):
        """普通重要性采样"""
        returns = defaultdict(list)
        importance_ratios = defaultdict(list)
        
        for episode in episodes:
            # 计算重要性采样比率
            rho = 1.0
            for state, action, reward in episode:
                s_idx = self.state_to_idx[state]
                a_idx = self.action_to_idx[action]
                
                pi_target = target_policy[s_idx, a_idx]
                pi_behavior = behavior_policy[s_idx, a_idx]
                
                if pi_behavior > 0:
                    rho *= pi_target / pi_behavior
                else:
                    rho = 0
                    break
            
            # 计算回报
            G = 0
            episode_returns = []
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                episode_returns.append(G)
            episode_returns.reverse()
            
            # 更新估计 (First-Visit)
            visited_states = set()
            for t, (state, action, reward) in enumerate(episode):
                if state not in visited_states:
                    visited_states.add(state)
                    returns[state].append(rho * episode_returns[t])
                    importance_ratios[state].append(rho)
        
        # 计算价值函数
        V = np.zeros(self.n_states)
        for state in self.states:
            if state in returns and len(returns[state]) > 0:
                V[self.state_to_idx[state]] = np.mean(returns[state])
        
        if verbose:
            print(f"✅ 普通重要性采样完成")
            for state in self.states:
                if state in returns:
                    avg_ratio = np.mean(importance_ratios[state])
                    print(f"   V({state}) = {V[self.state_to_idx[state]]:.4f}, 平均重要性比率 = {avg_ratio:.4f}")
        
        return V
    
    def _weighted_importance_sampling(self, episodes, behavior_policy, target_policy, verbose):
        """加权重要性采样"""
        weighted_returns = defaultdict(float)
        importance_weights = defaultdict(float)
        
        for episode in episodes:
            # 计算重要性采样比率
            rho = 1.0
            for state, action, reward in episode:
                s_idx = self.state_to_idx[state]
                a_idx = self.action_to_idx[action]
                
                pi_target = target_policy[s_idx, a_idx]
                pi_behavior = behavior_policy[s_idx, a_idx]
                
                if pi_behavior > 0:
                    rho *= pi_target / pi_behavior
                else:
                    rho = 0
                    break
            
            # 计算回报
            G = 0
            episode_returns = []
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                episode_returns.append(G)
            episode_returns.reverse()
            
            # 更新加权估计 (First-Visit)
            visited_states = set()
            for t, (state, action, reward) in enumerate(episode):
                if state not in visited_states:
                    visited_states.add(state)
                    weighted_returns[state] += rho * episode_returns[t]
                    importance_weights[state] += rho
        
        # 计算价值函数
        V = np.zeros(self.n_states)
        for state in self.states:
            if state in weighted_returns and importance_weights[state] > 0:
                V[self.state_to_idx[state]] = weighted_returns[state] / importance_weights[state]
        
        if verbose:
            print(f"✅ 加权重要性采样完成")
            for state in self.states:
                if state in weighted_returns:
                    weight = importance_weights[state]
                    print(f"   V({state}) = {V[self.state_to_idx[state]]:.4f}, 总权重 = {weight:.4f}")
        
        return V


class BellmanMonteCarloVisualizer:
    """贝尔曼期望方程与蒙特卡洛方法可视化类"""
    
    def __init__(self, bellman_solver: BellmanExpectationEquation, 
                 mc_methods: MonteCarloMethods):
        """
        初始化可视化器
        
        参数:
            bellman_solver: 贝尔曼方程求解器
            mc_methods: 蒙特卡洛方法实例
        """
        self.bellman = bellman_solver
        self.mc = mc_methods
        plt.style.use('default')
        # 重新设置中文字体，确保在样式设置后生效
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_bellman_equation_breakdown(self, policy: np.ndarray, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        绘制贝尔曼期望方程的分解图
        
        参数:
            policy: 策略矩阵
            figsize: 图形大小
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        
        # 1. 策略可视化
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(policy, cmap='Blues', aspect='auto')
        ax1.set_title('策略 π(a|s)')
        ax1.set_xlabel('动作')
        ax1.set_ylabel('状态')
        ax1.set_xticks(range(len(self.bellman.actions)))
        ax1.set_xticklabels(self.bellman.actions)
        ax1.set_yticks(range(len(self.bellman.states)))
        ax1.set_yticklabels(self.bellman.states)
        plt.colorbar(im1, ax=ax1)
        
        # 2. 即时奖励
        ax2 = fig.add_subplot(gs[0, 1])
        immediate_rewards = np.zeros((self.bellman.n_states, self.bellman.n_actions))
        for s in range(self.bellman.n_states):
            for a in range(self.bellman.n_actions):
                immediate_rewards[s, a] = np.sum(self.bellman.P[s, a, :] * self.bellman.R[s, a, :])
        
        im2 = ax2.imshow(immediate_rewards, cmap='RdYlGn', aspect='auto')
        ax2.set_title('即时奖励 E[R|s,a]')
        ax2.set_xlabel('动作')
        ax2.set_ylabel('状态')
        ax2.set_xticks(range(len(self.bellman.actions)))
        ax2.set_xticklabels(self.bellman.actions)
        ax2.set_yticks(range(len(self.bellman.states)))
        ax2.set_yticklabels(self.bellman.states)
        plt.colorbar(im2, ax=ax2)
        
        # 3. 状态价值函数
        ax3 = fig.add_subplot(gs[0, 2])
        V = self.bellman.solve_bellman_expectation_v(policy, method='iterative')
        bars = ax3.bar(self.bellman.states, V, color='skyblue', alpha=0.7)
        ax3.set_title('状态价值函数 V^π(s)')
        ax3.set_xlabel('状态')
        ax3.set_ylabel('价值')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标注
        for bar, value in zip(bars, V):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 4. 贝尔曼方程分解
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # 显示贝尔曼方程的数学形式和数值计算
        equation_text = """
        贝尔曼期望方程分解：
        
        V^π(s) = Σ_a π(a|s) × Σ_{s'} P(s'|s,a) × [R(s,a,s') + γ × V^π(s')]
                 ↑           ↑                      ↑              ↑
              策略概率    转移概率              即时奖励      折扣未来价值
        
        数值示例 (以第一个状态为例):
        """
        
        # 计算第一个状态的贝尔曼方程分解
        s = 0
        breakdown_text = f"\n        V^π({self.bellman.states[s]}) = "
        total_value = 0
        
        for a in range(self.bellman.n_actions):
            pi_sa = policy[s, a]
            if pi_sa > 0.001:  # 只显示有意义的项
                action_value = 0
                for s_next in range(self.bellman.n_states):
                    p_sas = self.bellman.P[s, a, s_next]
                    r_sas = self.bellman.R[s, a, s_next]
                    if p_sas > 0:
                        action_value += p_sas * (r_sas + self.bellman.gamma * V[s_next])
                
                term_value = pi_sa * action_value
                total_value += term_value
                breakdown_text += f"\n                     + {pi_sa:.3f} × {action_value:.3f} ({self.bellman.actions[a]})"
        
        breakdown_text += f"\n                   = {total_value:.3f}"
        
        equation_text += breakdown_text
        
        ax4.text(0.05, 0.95, equation_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('贝尔曼期望方程详细分解', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_mc_convergence_analysis(self, true_values: np.ndarray,
                                   mc_first_visit: np.ndarray,
                                   mc_every_visit: np.ndarray,
                                   figsize: Tuple[int, int] = (15, 8)):
        """
        绘制蒙特卡洛方法收敛分析
        
        参数:
            true_values: 真实价值函数
            mc_first_visit: First-Visit MC估计
            mc_every_visit: Every-Visit MC估计
            figsize: 图形大小
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 价值函数比较
        x = np.arange(len(self.mc.states))
        width = 0.25
        
        axes[0, 0].bar(x - width, true_values, width, label='真实值', alpha=0.8, color='green')
        axes[0, 0].bar(x, mc_first_visit, width, label='First-Visit MC', alpha=0.8, color='blue')
        axes[0, 0].bar(x + width, mc_every_visit, width, label='Every-Visit MC', alpha=0.8, color='red')
        
        axes[0, 0].set_xlabel('状态')
        axes[0, 0].set_ylabel('价值')
        axes[0, 0].set_title('价值函数估计比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.mc.states, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差分析
        error_first = np.abs(mc_first_visit - true_values)
        error_every = np.abs(mc_every_visit - true_values)
        
        axes[0, 1].bar(x - width/2, error_first, width, label='First-Visit 误差', alpha=0.8, color='blue')
        axes[0, 1].bar(x + width/2, error_every, width, label='Every-Visit 误差', alpha=0.8, color='red')
        
        axes[0, 1].set_xlabel('状态')
        axes[0, 1].set_ylabel('绝对误差')
        axes[0, 1].set_title('估计误差比较')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.mc.states, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 学习曲线 (如果有历史数据)
        if hasattr(self.mc, 'episode_returns') and self.mc.episode_returns:
            episodes = range(len(self.mc.episode_returns))
            returns = self.mc.episode_returns
            
            # 计算移动平均
            window_size = min(50, len(returns) // 10)
            if window_size > 1:
                moving_avg = []
                for i in range(len(returns)):
                    start = max(0, i - window_size + 1)
                    moving_avg.append(np.mean(returns[start:i+1]))
                
                axes[1, 0].plot(episodes, returns, alpha=0.3, color='gray', label='回合奖励')
                axes[1, 0].plot(episodes, moving_avg, color='blue', linewidth=2, label=f'{window_size}回合移动平均')
            else:
                axes[1, 0].plot(episodes, returns, color='blue', label='回合奖励')
            
            axes[1, 0].set_xlabel('回合')
            axes[1, 0].set_ylabel('回合奖励')
            axes[1, 0].set_title('学习曲线')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '无学习历史数据', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('学习曲线')
        
        # 4. 统计摘要
        axes[1, 1].axis('off')
        
        # 计算统计指标
        mse_first = np.mean((mc_first_visit - true_values) ** 2)
        mse_every = np.mean((mc_every_visit - true_values) ** 2)
        mae_first = np.mean(np.abs(mc_first_visit - true_values))
        mae_every = np.mean(np.abs(mc_every_visit - true_values))
        
        stats_text = f"""
        蒙特卡洛方法性能比较
        ═══════════════════════════════════════
        
        First-Visit MC:
        • 均方误差 (MSE): {mse_first:.6f}
        • 平均绝对误差 (MAE): {mae_first:.6f}
        • 最大误差: {np.max(error_first):.6f}
        
        Every-Visit MC:
        • 均方误差 (MSE): {mse_every:.6f}
        • 平均绝对误差 (MAE): {mae_every:.6f}
        • 最大误差: {np.max(error_every):.6f}
        
        方法比较:
        • 更优方法: {"First-Visit" if mse_first < mse_every else "Every-Visit"}
        • MSE改进: {abs(mse_first - mse_every):.6f}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('蒙特卡洛方法收敛分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_importance_sampling_comparison(self, ordinary_is: np.ndarray,
                                          weighted_is: np.ndarray,
                                          true_values: np.ndarray,
                                          figsize: Tuple[int, int] = (12, 8)):
        """
        绘制重要性采样方法比较
        
        参数:
            ordinary_is: 普通重要性采样结果
            weighted_is: 加权重要性采样结果
            true_values: 真实价值函数
            figsize: 图形大小
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 价值函数比较
        x = np.arange(len(self.mc.states))
        width = 0.2
        
        axes[0, 0].bar(x - width, true_values, width, label='真实值', alpha=0.8, color='green')
        axes[0, 0].bar(x, ordinary_is, width, label='普通重要性采样', alpha=0.8, color='blue')
        axes[0, 0].bar(x + width, weighted_is, width, label='加权重要性采样', alpha=0.8, color='red')
        
        axes[0, 0].set_xlabel('状态')
        axes[0, 0].set_ylabel('价值')
        axes[0, 0].set_title('重要性采样方法比较')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.mc.states, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差分析
        error_ordinary = np.abs(ordinary_is - true_values)
        error_weighted = np.abs(weighted_is - true_values)
        
        axes[0, 1].bar(x - width/2, error_ordinary, width, label='普通IS误差', alpha=0.8, color='blue')
        axes[0, 1].bar(x + width/2, error_weighted, width, label='加权IS误差', alpha=0.8, color='red')
        
        axes[0, 1].set_xlabel('状态')
        axes[0, 1].set_ylabel('绝对误差')
        axes[0, 1].set_title('重要性采样误差比较')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.mc.states, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 方差分析 (模拟)
        # 这里我们模拟不同方法的方差特性
        episodes_range = np.arange(10, 1001, 50)
        ordinary_variance = 1.0 / np.sqrt(episodes_range)  # 普通IS方差较高
        weighted_variance = 0.5 / np.sqrt(episodes_range)  # 加权IS方差较低
        
        axes[1, 0].plot(episodes_range, ordinary_variance, 'b-', label='普通重要性采样', linewidth=2)
        axes[1, 0].plot(episodes_range, weighted_variance, 'r-', label='加权重要性采样', linewidth=2)
        axes[1, 0].set_xlabel('回合数')
        axes[1, 0].set_ylabel('估计方差')
        axes[1, 0].set_title('方差收敛特性')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 4. 统计摘要
        axes[1, 1].axis('off')
        
        mse_ordinary = np.mean((ordinary_is - true_values) ** 2)
        mse_weighted = np.mean((weighted_is - true_values) ** 2)
        
        stats_text = f"""
        重要性采样方法性能比较
        ═══════════════════════════════════════
        
        普通重要性采样:
        • 均方误差: {mse_ordinary:.6f}
        • 平均绝对误差: {np.mean(error_ordinary):.6f}
        • 最大误差: {np.max(error_ordinary):.6f}
        • 特点: 无偏估计，但方差可能很大
        
        加权重要性采样:
        • 均方误差: {mse_weighted:.6f}
        • 平均绝对误差: {np.mean(error_weighted):.6f}
        • 最大误差: {np.max(error_weighted):.6f}
        • 特点: 有偏估计，但方差较小
        
        推荐方法: {"加权重要性采样" if mse_weighted < mse_ordinary else "普通重要性采样"}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('重要性采样方法比较分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def create_simple_mdp_example():
    """创建简单的MDP示例用于演示"""
    print("\n" + "="*80)
    print("🎯 创建简单MDP示例：学生学习决策过程")
    print("="*80)
    
    states = ['困倦', '清醒', '专注']
    actions = ['休息', '学习']
    
    # 转移概率 {(s, a, s'): P(s'|s,a)}
    transition_probs = {
        ('困倦', '休息', '困倦'): 0.3,
        ('困倦', '休息', '清醒'): 0.7,
        ('困倦', '学习', '困倦'): 0.8,
        ('困倦', '学习', '清醒'): 0.2,
        
        ('清醒', '休息', '困倦'): 0.4,
        ('清醒', '休息', '清醒'): 0.6,
        ('清醒', '学习', '清醒'): 0.4,
        ('清醒', '学习', '专注'): 0.6,
        
        ('专注', '休息', '清醒'): 0.8,
        ('专注', '休息', '专注'): 0.2,
        ('专注', '学习', '专注'): 0.9,
        ('专注', '学习', '清醒'): 0.1,
    }
    
    # 奖励函数 {(s, a, s'): R(s,a,s')}
    rewards = {
        ('困倦', '休息', '困倦'): -1,
        ('困倦', '休息', '清醒'): 2,
        ('困倦', '学习', '困倦'): -5,
        ('困倦', '学习', '清醒'): 1,
        
        ('清醒', '休息', '困倦'): -2,
        ('清醒', '休息', '清醒'): 0,
        ('清醒', '学习', '清醒'): 3,
        ('清醒', '学习', '专注'): 8,
        
        ('专注', '休息', '清醒'): -3,
        ('专注', '休息', '专注'): 1,
        ('专注', '学习', '专注'): 10,
        ('专注', '学习', '清醒'): 5,
    }
    
    return states, actions, transition_probs, rewards


def demonstrate_bellman_monte_carlo():
    """演示贝尔曼期望方程与蒙特卡洛方法"""
    print("\n" + "🧠" * 40)
    print("贝尔曼期望方程与蒙特卡洛方法核心概念演示")
    print("🧠" * 40)
    
    # 创建示例
    states, actions, transition_probs, rewards = create_simple_mdp_example()
    
    # 初始化求解器
    bellman_solver = BellmanExpectationEquation(states, actions, transition_probs, rewards)
    mc_methods = MonteCarloMethods(states, actions)
    
    # 创建测试策略
    print("\n📋 创建测试策略")
    policy = np.array([
        [0.7, 0.3],  # 困倦: 70%休息, 30%学习
        [0.4, 0.6],  # 清醒: 40%休息, 60%学习
        [0.2, 0.8],  # 专注: 20%休息, 80%学习
    ])
    
    print("策略设置:")
    for i, state in enumerate(states):
        print(f"  {state}: {policy[i, 0]:.1%}休息, {policy[i, 1]:.1%}学习")
    
    # 1. 求解贝尔曼期望方程
    print("\n" + "="*60)
    print("🧮 第一步：求解贝尔曼期望方程")
    print("="*60)
    
    V_analytical = bellman_solver.solve_bellman_expectation_v(policy, method='analytical')
    V_iterative = bellman_solver.solve_bellman_expectation_v(policy, method='iterative')
    Q_function = bellman_solver.solve_bellman_expectation_q(policy)
    
    print(f"\n价值函数比较:")
    print(f"{'状态':<8} {'解析解':<10} {'迭代解':<10} {'差异':<10}")
    print("-" * 40)
    for i, state in enumerate(states):
        diff = abs(V_analytical[i] - V_iterative[i])
        print(f"{state:<8} {V_analytical[i]:<10.4f} {V_iterative[i]:<10.4f} {diff:<10.6f}")
    
    # 2. 生成模拟数据
    print("\n" + "="*60)
    print("🎲 第二步：生成蒙特卡洛模拟数据")
    print("="*60)
    
    def simple_environment_simulator(num_episodes=500, max_steps=20):
        """简单环境模拟器"""
        episodes = []
        
        for _ in range(num_episodes):
            episode = []
            state = np.random.choice(states)  # 随机起始状态
            
            for _ in range(max_steps):
                # 根据策略选择动作
                state_idx = bellman_solver.state_to_idx[state]
                action_probs = policy[state_idx]
                action_idx = np.random.choice(len(actions), p=action_probs)
                action = actions[action_idx]
                
                # 状态转移和奖励
                next_state_probs = bellman_solver.P[state_idx, action_idx]
                next_state_idx = np.random.choice(len(states), p=next_state_probs)
                next_state = states[next_state_idx]
                
                reward = bellman_solver.R[state_idx, action_idx, next_state_idx]
                
                episode.append((state, action, reward))
                state = next_state
            
            episodes.append(episode)
        
        return episodes
    
    episodes = simple_environment_simulator(num_episodes=1000)
    print(f"生成了 {len(episodes)} 个回合的模拟数据")
    print(f"平均回合长度: {np.mean([len(ep) for ep in episodes]):.1f}")
    
    # 3. 蒙特卡洛预测
    print("\n" + "="*60)
    print("🎯 第三步：蒙特卡洛预测方法")
    print("="*60)
    
    V_first_visit = mc_methods.first_visit_mc_prediction(episodes)
    V_every_visit = mc_methods.every_visit_mc_prediction(episodes)
    
    # 4. 重要性采样演示
    print("\n" + "="*60)
    print("🎭 第四步：重要性采样方法")
    print("="*60)
    
    # 创建行为策略（更随机）
    behavior_policy = np.array([
        [0.5, 0.5],  # 均匀策略
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    
    # 生成行为策略的数据
    def behavior_simulator(num_episodes=300):
        episodes = []
        for _ in range(num_episodes):
            episode = []
            state = np.random.choice(states)
            
            for _ in range(15):
                state_idx = bellman_solver.state_to_idx[state]
                action_idx = np.random.choice(len(actions), p=behavior_policy[state_idx])
                action = actions[action_idx]
                
                next_state_probs = bellman_solver.P[state_idx, action_idx]
                next_state_idx = np.random.choice(len(states), p=next_state_probs)
                next_state = states[next_state_idx]
                
                reward = bellman_solver.R[state_idx, action_idx, next_state_idx]
                episode.append((state, action, reward))
                state = next_state
            
            episodes.append(episode)
        return episodes
    
    behavior_episodes = behavior_simulator()
    
    V_ordinary_is = mc_methods.importance_sampling_prediction(
        behavior_episodes, behavior_policy, policy, method='ordinary')
    V_weighted_is = mc_methods.importance_sampling_prediction(
        behavior_episodes, behavior_policy, policy, method='weighted')
    
    # 5. 可视化分析
    print("\n" + "="*60)
    print("📈 第五步：生成可视化分析")
    print("="*60)
    
    visualizer = BellmanMonteCarloVisualizer(bellman_solver, mc_methods)
    
    print("1. 贝尔曼期望方程分解图")
    visualizer.plot_bellman_equation_breakdown(policy)
    
    print("2. 蒙特卡洛方法收敛分析")
    visualizer.plot_mc_convergence_analysis(V_analytical, V_first_visit, V_every_visit)
    
    print("3. 重要性采样方法比较")
    visualizer.plot_importance_sampling_comparison(V_ordinary_is, V_weighted_is, V_analytical)
    
    # 6. 最终总结
    print("\n" + "="*80)
    print("📋 贝尔曼期望方程与蒙特卡洛方法分析总结")
    print("="*80)
    
    print(f"\n🧮 贝尔曼期望方程结果:")
    print(f"  解析解与迭代解最大差异: {np.max(np.abs(V_analytical - V_iterative)):.8f}")
    
    print(f"\n🎲 蒙特卡洛预测结果:")
    mc_first_error = np.mean(np.abs(V_first_visit - V_analytical))
    mc_every_error = np.mean(np.abs(V_every_visit - V_analytical))
    print(f"  First-Visit MC 平均误差: {mc_first_error:.4f}")
    print(f"  Every-Visit MC 平均误差: {mc_every_error:.4f}")
    print(f"  更优方法: {'First-Visit' if mc_first_error < mc_every_error else 'Every-Visit'}")
    
    print(f"\n🎭 重要性采样结果:")
    is_ordinary_error = np.mean(np.abs(V_ordinary_is - V_analytical))
    is_weighted_error = np.mean(np.abs(V_weighted_is - V_analytical))
    print(f"  普通重要性采样平均误差: {is_ordinary_error:.4f}")
    print(f"  加权重要性采样平均误差: {is_weighted_error:.4f}")
    print(f"  更优方法: {'普通重要性采样' if is_ordinary_error < is_weighted_error else '加权重要性采样'}")
    
    print(f"\n🎯 核心概念总结:")
    print(f"  1. 贝尔曼期望方程提供了价值函数的递归定义")
    print(f"  2. 蒙特卡洛方法通过采样估计价值函数，无需模型")
    print(f"  3. 重要性采样允许使用不同策略的数据进行学习")
    print(f"  4. 加权重要性采样通常比普通重要性采样更稳定")


if __name__ == "__main__":
    """主程序入口"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("🚀 快速演示模式")
        print("="*50)
        
        # 快速演示核心概念
        states, actions, transition_probs, rewards = create_simple_mdp_example()
        bellman_solver = BellmanExpectationEquation(states, actions, transition_probs, rewards)
        
        # 简单策略
        policy = np.array([[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
        
        # 求解贝尔曼方程
        V = bellman_solver.solve_bellman_expectation_v(policy, method='iterative')
        
        print(f"\n状态价值函数:")
        for i, state in enumerate(states):
            print(f"  V({state}) = {V[i]:.4f}")
        
        print(f"\n✅ 快速演示完成！运行 'python bellman_monte_carlo.py' 查看完整演示")
    
    else:
        # 完整演示
        demonstrate_bellman_monte_carlo()