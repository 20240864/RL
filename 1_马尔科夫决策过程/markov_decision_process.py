"""
马尔科夫决策过程 (Markov Decision Process, MDP) 详细实现

本文件提供了MDP的完整实现，包括：
1. 核心MDP数学理论的严格实现
2. 策略表示和评估
3. 策略迭代和价值迭代算法
4. 丰富的可视化功能
5. 详细的教学示例和分析

作者：强化学习教学代码
版本：2.0
用途：深入理解马尔科夫决策过程的数学原理和算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MarkovDecisionProcess:
    """
    马尔科夫决策过程详细实现类
    
    马尔科夫决策过程是一个五元组 (S, A, P, R, γ)：
    - S: 状态空间 (State Space)
    - A: 动作空间 (Action Space)
    - P: 状态转移概率函数 P(s'|s,a)
    - R: 奖励函数 R(s,a,s') 或 R(s,a)
    - γ: 折扣因子 (Discount Factor)
    
    核心概念：
    1. 策略 π(a|s): 在状态s下选择动作a的概率
    2. 状态价值函数 V^π(s): 在策略π下状态s的期望回报
    3. 动作价值函数 Q^π(s,a): 在策略π下状态s执行动作a的期望回报
    4. 贝尔曼方程：V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ*V^π(s')]
    """
    
    def __init__(self, states: List[str], actions: List[str], 
                 transition_probs: Dict[Tuple[str, str, str], float],
                 rewards: Dict[Tuple[str, str], float], gamma: float,
                 state_descriptions: Optional[Dict[str, str]] = None,
                 action_descriptions: Optional[Dict[str, str]] = None):
        """
        初始化马尔科夫决策过程
        
        参数:
            states: 状态名称列表
            actions: 动作名称列表
            transition_probs: 转移概率字典 {(s, a, s'): P(s'|s,a)}
            rewards: 奖励函数字典 {(s, a): R(s,a)}
            gamma: 折扣因子 γ ∈ [0,1]
            state_descriptions: 状态描述字典（可选）
            action_descriptions: 动作描述字典（可选）
        """
        self.states = states
        self.actions = actions
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.state_to_index = {state: i for i, state in enumerate(states)}
        self.action_to_index = {action: i for i, action in enumerate(actions)}
        self.index_to_state = {i: state for i, state in enumerate(states)}
        self.index_to_action = {i: action for i, action in enumerate(actions)}
        
        self.gamma = gamma
        self.state_descriptions = state_descriptions or {}
        self.action_descriptions = action_descriptions or {}
        
        # 构建转移概率张量和奖励矩阵
        self._build_transition_tensor(transition_probs)
        self._build_reward_matrix(rewards)
        
        # 验证输入参数
        self._validate_parameters()
        
        # 存储计算历史
        self.policy_iteration_history = []
        self.value_iteration_history = []
        self.current_policy = None
        self.current_value_function = None
        
        print(f"✅ 马尔科夫决策过程初始化完成")
        print(f"   状态数量: {self.n_states}")
        print(f"   动作数量: {self.n_actions}")
        print(f"   折扣因子: {self.gamma}")
        print(f"   状态列表: {self.states}")
        print(f"   动作列表: {self.actions}")
    
    def _build_transition_tensor(self, transition_probs: Dict[Tuple[str, str, str], float]):
        """构建转移概率张量 P[s,a,s'] = P(s'|s,a)"""
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for (s, a, s_next), prob in transition_probs.items():
            s_idx = self.state_to_index[s]
            a_idx = self.action_to_index[a]
            s_next_idx = self.state_to_index[s_next]
            self.P[s_idx, a_idx, s_next_idx] = prob
    
    def _build_reward_matrix(self, rewards: Dict[Tuple[str, str], float]):
        """构建奖励矩阵 R[s,a] = R(s,a)"""
        self.R = np.zeros((self.n_states, self.n_actions))
        
        for (s, a), reward in rewards.items():
            s_idx = self.state_to_index[s]
            a_idx = self.action_to_index[a]
            self.R[s_idx, a_idx] = reward
    
    def _validate_parameters(self):
        """验证输入参数的有效性"""
        # 检查转移概率张量
        for s_idx in range(self.n_states):
            for a_idx in range(self.n_actions):
                prob_sum = np.sum(self.P[s_idx, a_idx, :])
                if not np.isclose(prob_sum, 1.0, rtol=1e-10):
                    state = self.states[s_idx]
                    action = self.actions[a_idx]
                    print(f"⚠️  警告: 状态 {state} 动作 {action} 的转移概率和为 {prob_sum:.6f}")
        
        # 检查概率非负
        if np.any(self.P < 0):
            raise ValueError("转移概率不能为负数")
        
        # 检查折扣因子
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"折扣因子必须在[0,1]范围内, 实际: {self.gamma}")
    
    def create_random_policy(self) -> np.ndarray:
        """
        创建随机策略（均匀分布）
        
        返回:
            策略矩阵 π[s,a] = π(a|s)
        """
        policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        return policy
    
    def create_deterministic_policy(self, action_mapping: Dict[str, str]) -> np.ndarray:
        """
        创建确定性策略
        
        参数:
            action_mapping: 状态到动作的映射 {state: action}
            
        返回:
            策略矩阵 π[s,a] = π(a|s)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for state, action in action_mapping.items():
            s_idx = self.state_to_index[state]
            a_idx = self.action_to_index[action]
            policy[s_idx, a_idx] = 1.0
        
        return policy
    
    def policy_evaluation(self, policy: np.ndarray, max_iterations: int = 1000, 
                         tolerance: float = 1e-8, verbose: bool = True) -> np.ndarray:
        """
        策略评估：计算给定策略的状态价值函数
        
        使用迭代方法求解贝尔曼期望方程：
        V^π(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a) + γ*V^π(s')]
        
        参数:
            policy: 策略矩阵 π[s,a]
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否显示详细信息
            
        返回:
            状态价值函数 V^π
        """
        if verbose:
            print(f"\n🔍 策略评估开始")
            print(f"   最大迭代次数: {max_iterations}")
            print(f"   收敛容忍度: {tolerance}")
        
        V = np.zeros(self.n_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.n_states):
                v_new = 0
                for a in range(self.n_actions):
                    action_value = 0
                    for s_next in range(self.n_states):
                        action_value += self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V_old[s_next])
                    v_new += policy[s, a] * action_value
                V[s] = v_new
            
            # 检查收敛
            delta = np.max(np.abs(V - V_old))
            
            if verbose and (iteration < 10 or iteration % 100 == 0):
                print(f"   迭代 {iteration:3d}: 最大变化量 = {delta:.8f}")
            
            if delta < tolerance:
                if verbose:
                    print(f"   ✅ 在第 {iteration} 次迭代后收敛")
                break
        else:
            if verbose:
                print(f"   ⚠️  达到最大迭代次数，未完全收敛")
        
        return V
    
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        策略改进：基于价值函数生成贪婪策略
        
        新策略：π'(s) = argmax_a Σ_{s'} P(s'|s,a) * [R(s,a) + γ*V(s')]
        
        参数:
            V: 状态价值函数
            
        返回:
            (新策略, 是否稳定)
        """
        new_policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for s in range(self.n_states):
            # 计算每个动作的Q值
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    q_values[a] += self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
            
            # 选择最优动作（贪婪策略）
            best_actions = np.where(q_values == np.max(q_values))[0]
            
            # 如果有多个最优动作，平均分配概率
            for a in best_actions:
                new_policy[s, a] = 1.0 / len(best_actions)
        
        return new_policy, policy_stable
    
    def policy_iteration(self, initial_policy: Optional[np.ndarray] = None, 
                        max_iterations: int = 100, tolerance: float = 1e-8,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法
        
        交替进行策略评估和策略改进，直到策略收敛
        
        参数:
            initial_policy: 初始策略（默认为随机策略）
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否显示详细信息
            
        返回:
            (最优策略, 最优价值函数)
        """
        if verbose:
            print(f"\n🔄 策略迭代算法开始")
            print("=" * 60)
        
        # 初始化策略
        if initial_policy is None:
            policy = self.create_random_policy()
        else:
            policy = initial_policy.copy()
        
        self.policy_iteration_history = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n策略迭代 - 第 {iteration + 1} 轮:")
            
            # 策略评估
            V = self.policy_evaluation(policy, verbose=False)
            
            # 策略改进
            new_policy, policy_stable = self.policy_improvement(V)
            
            # 检查策略变化
            policy_change = np.max(np.abs(new_policy - policy))
            self.policy_iteration_history.append({
                'iteration': iteration,
                'policy_change': policy_change,
                'value_function': V.copy(),
                'policy': new_policy.copy()
            })
            
            if verbose:
                print(f"   策略变化量: {policy_change:.8f}")
                print(f"   价值函数: {V}")
            
            # 检查收敛
            if policy_change < tolerance:
                if verbose:
                    print(f"   ✅ 策略在第 {iteration + 1} 轮后收敛")
                break
            
            policy = new_policy
        else:
            if verbose:
                print(f"   ⚠️  达到最大迭代次数，策略可能未完全收敛")
        
        self.current_policy = policy
        self.current_value_function = V
        
        if verbose:
            print(f"\n🎯 最优策略:")
            self._print_policy(policy)
        
        return policy, V
    
    def value_iteration(self, max_iterations: int = 1000, tolerance: float = 1e-8,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        价值迭代算法
        
        直接迭代贝尔曼最优方程：
        V*(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a) + γ*V*(s')]
        
        参数:
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否显示详细信息
            
        返回:
            (最优策略, 最优价值函数)
        """
        if verbose:
            print(f"\n⚡ 价值迭代算法开始")
            print("=" * 60)
            print(f"   最大迭代次数: {max_iterations}")
            print(f"   收敛容忍度: {tolerance}")
        
        V = np.zeros(self.n_states)
        self.value_iteration_history = []
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s in range(self.n_states):
                # 计算所有动作的Q值
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        q_values[a] += self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V_old[s_next])
                
                # 选择最大Q值
                V[s] = np.max(q_values)
            
            # 计算变化量
            delta = np.max(np.abs(V - V_old))
            self.value_iteration_history.append({
                'iteration': iteration,
                'delta': delta,
                'value_function': V.copy()
            })
            
            if verbose and (iteration < 10 or iteration % 100 == 0):
                print(f"   迭代 {iteration:3d}: 最大变化量 = {delta:.8f}")
            
            # 检查收敛
            if delta < tolerance:
                if verbose:
                    print(f"   ✅ 在第 {iteration} 次迭代后收敛")
                break
        else:
            if verbose:
                print(f"   ⚠️  达到最大迭代次数，未完全收敛")
        
        # 提取最优策略
        optimal_policy, _ = self.policy_improvement(V)
        
        self.current_policy = optimal_policy
        self.current_value_function = V
        
        if verbose:
            print(f"\n🎯 最优策略:")
            self._print_policy(optimal_policy)
            print(f"\n💰 最优价值函数:")
            for i, (state, value) in enumerate(zip(self.states, V)):
                print(f"   V*({state}) = {value:.6f}")
        
        return optimal_policy, V
    
    def compute_q_function(self, policy: np.ndarray) -> np.ndarray:
        """
        计算动作价值函数 Q^π(s,a)
        
        Q^π(s,a) = Σ_{s'} P(s'|s,a) * [R(s,a) + γ*V^π(s')]
        
        参数:
            policy: 策略矩阵
            
        返回:
            动作价值函数 Q[s,a]
        """
        V = self.policy_evaluation(policy, verbose=False)
        Q = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    Q[s, a] += self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
        
        return Q
    
    def simulate_episode(self, policy: np.ndarray, start_state: str, 
                        max_steps: int = 100, random_seed: Optional[int] = None) -> Tuple[List[str], List[str], List[float]]:
        """
        根据给定策略模拟一个回合
        
        参数:
            policy: 策略矩阵
            start_state: 起始状态
            max_steps: 最大步数
            random_seed: 随机种子
            
        返回:
            (状态序列, 动作序列, 奖励序列)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        states_sequence = [start_state]
        actions_sequence = []
        rewards_sequence = []
        
        current_state_idx = self.state_to_index[start_state]
        
        for step in range(max_steps):
            # 根据策略选择动作
            action_probs = policy[current_state_idx]
            action_idx = np.random.choice(self.n_actions, p=action_probs)
            action = self.actions[action_idx]
            actions_sequence.append(action)
            
            # 获得奖励
            reward = self.R[current_state_idx, action_idx]
            rewards_sequence.append(reward)
            
            # 状态转移
            next_state_probs = self.P[current_state_idx, action_idx]
            next_state_idx = np.random.choice(self.n_states, p=next_state_probs)
            next_state = self.states[next_state_idx]
            states_sequence.append(next_state)
            
            current_state_idx = next_state_idx
        
        return states_sequence, actions_sequence, rewards_sequence
    
    def _print_policy(self, policy: np.ndarray):
        """打印策略的可读格式"""
        print("   状态 -> 动作概率分布:")
        for s_idx, state in enumerate(self.states):
            action_probs = policy[s_idx]
            prob_str = ", ".join([f"{action}:{prob:.3f}" for action, prob in zip(self.actions, action_probs) if prob > 0.001])
            print(f"     {state}: {prob_str}")
    
    def print_detailed_summary(self):
        """打印详细的MDP信息摘要"""
        print("\n" + "=" * 80)
        print("马尔科夫决策过程 (MDP) 详细信息摘要")
        print("=" * 80)
        
        print(f"\n📋 基本信息:")
        print(f"   状态空间大小: {self.n_states}")
        print(f"   动作空间大小: {self.n_actions}")
        print(f"   状态列表: {self.states}")
        print(f"   动作列表: {self.actions}")
        print(f"   折扣因子 γ: {self.gamma}")
        
        if self.state_descriptions:
            print(f"\n📝 状态描述:")
            for state, desc in self.state_descriptions.items():
                print(f"   {state}: {desc}")
        
        if self.action_descriptions:
            print(f"\n🎮 动作描述:")
            for action, desc in self.action_descriptions.items():
                print(f"   {action}: {desc}")
        
        print(f"\n🎯 奖励函数 R(s,a):")
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                reward = self.R[s_idx, a_idx]
                if reward != 0:  # 只显示非零奖励
                    print(f"   R({state}, {action}) = {reward:8.3f}")
        
        print(f"\n🔄 状态转移概率 P(s'|s,a) (只显示非零概率):")
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                transitions = []
                for s_next_idx, next_state in enumerate(self.states):
                    prob = self.P[s_idx, a_idx, s_next_idx]
                    if prob > 0.001:
                        transitions.append(f"{next_state}:{prob:.3f}")
                if transitions:
                    print(f"   P(·|{state},{action}) = {{{', '.join(transitions)}}}")


class MDPVisualizer:
    """马尔科夫决策过程可视化类"""
    
    def __init__(self, mdp: MarkovDecisionProcess):
        """
        初始化可视化器
        
        参数:
            mdp: 马尔科夫决策过程实例
        """
        self.mdp = mdp
        plt.style.use('default')
        # 重新设置中文字体，确保在style设置后生效
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_policy_visualization(self, policy: np.ndarray, title: str = "策略可视化", 
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        绘制策略可视化图
        
        参数:
            policy: 策略矩阵
            title: 图标题
            figsize: 图形大小
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：策略热力图
        im1 = ax1.imshow(policy, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(self.mdp.n_actions))
        ax1.set_yticks(range(self.mdp.n_states))
        ax1.set_xticklabels(self.mdp.actions)
        ax1.set_yticklabels(self.mdp.states)
        ax1.set_xlabel('动作')
        ax1.set_ylabel('状态')
        ax1.set_title('策略概率矩阵 π(a|s)')
        
        # 添加数值标注
        for i in range(self.mdp.n_states):
            for j in range(self.mdp.n_actions):
                prob = policy[i, j]
                color = 'white' if prob > 0.5 else 'black'
                ax1.text(j, i, f'{prob:.2f}', ha='center', va='center', 
                        color=color, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='选择概率')
        
        # 右图：确定性策略图（显示每个状态的最优动作）
        deterministic_actions = np.argmax(policy, axis=1)
        colors = plt.cm.Set3(np.linspace(0, 1, self.mdp.n_actions))
        
        bars = ax2.bar(range(self.mdp.n_states), [1] * self.mdp.n_states, 
                      color=[colors[a] for a in deterministic_actions])
        
        ax2.set_xticks(range(self.mdp.n_states))
        ax2.set_xticklabels(self.mdp.states, rotation=45)
        ax2.set_ylabel('最优动作')
        ax2.set_title('确定性策略 (最优动作)')
        ax2.set_ylim(0, 1.2)
        
        # 添加动作标签
        for i, (bar, action_idx) in enumerate(zip(bars, deterministic_actions)):
            action = self.mdp.actions[action_idx]
            ax2.text(bar.get_x() + bar.get_width()/2., 0.5, action,
                    ha='center', va='center', fontweight='bold', rotation=90)
        
        # 添加图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=action) 
                          for i, action in enumerate(self.mdp.actions)]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_value_function(self, V: np.ndarray, title: str = "状态价值函数", 
                           figsize: Tuple[int, int] = (10, 6)):
        """
        绘制状态价值函数
        
        参数:
            V: 状态价值函数
            title: 图标题
            figsize: 图形大小
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建颜色映射
        colors = ['red' if v < 0 else 'green' if v > 0 else 'gray' for v in V]
        
        bars = ax.bar(self.mdp.states, V, color=colors, alpha=0.7)
        
        # 添加数值标注
        for bar, value in zip(bars, V):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        ax.set_xlabel('状态')
        ax.set_ylabel('价值函数 V(s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_q_function(self, Q: np.ndarray, title: str = "动作价值函数", 
                       figsize: Tuple[int, int] = (12, 8)):
        """
        绘制动作价值函数热力图
        
        参数:
            Q: 动作价值函数矩阵
            title: 图标题
            figsize: 图形大小
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建热力图
        im = ax.imshow(Q, cmap='RdYlGn', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(self.mdp.n_actions))
        ax.set_yticks(range(self.mdp.n_states))
        ax.set_xticklabels(self.mdp.actions)
        ax.set_yticklabels(self.mdp.states)
        ax.set_xlabel('动作')
        ax.set_ylabel('状态')
        ax.set_title(title + ' Q(s,a)')
        
        # 添加数值标注
        for i in range(self.mdp.n_states):
            for j in range(self.mdp.n_actions):
                value = Q[i, j]
                color = 'white' if abs(value) > np.max(np.abs(Q)) * 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q值', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_history(self, figsize: Tuple[int, int] = (15, 6)):
        """
        绘制算法收敛历史
        
        参数:
            figsize: 图形大小
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 策略迭代收敛历史
        if self.mdp.policy_iteration_history:
            iterations = [h['iteration'] for h in self.mdp.policy_iteration_history]
            policy_changes = [h['policy_change'] for h in self.mdp.policy_iteration_history]
            
            axes[0].semilogy(iterations, policy_changes, 'bo-', linewidth=2, markersize=6)
            axes[0].set_xlabel('迭代次数')
            axes[0].set_ylabel('策略变化量 (对数尺度)')
            axes[0].set_title('策略迭代收敛历史')
            axes[0].grid(True, alpha=0.3)
            
            if policy_changes:
                final_change = policy_changes[-1]
                axes[0].axhline(y=final_change, color='red', linestyle='--', alpha=0.7)
                axes[0].text(len(iterations)*0.7, final_change*2, 
                           f'最终变化: {final_change:.2e}', fontsize=10)
        else:
            axes[0].text(0.5, 0.5, '无策略迭代历史', ha='center', va='center', 
                        transform=axes[0].transAxes, fontsize=12)
            axes[0].set_title('策略迭代收敛历史')
        
        # 价值迭代收敛历史
        if self.mdp.value_iteration_history:
            iterations = [h['iteration'] for h in self.mdp.value_iteration_history]
            deltas = [h['delta'] for h in self.mdp.value_iteration_history]
            
            axes[1].semilogy(iterations, deltas, 'ro-', linewidth=2, markersize=6)
            axes[1].set_xlabel('迭代次数')
            axes[1].set_ylabel('最大变化量 (对数尺度)')
            axes[1].set_title('价值迭代收敛历史')
            axes[1].grid(True, alpha=0.3)
            
            if deltas:
                final_delta = deltas[-1]
                axes[1].axhline(y=final_delta, color='red', linestyle='--', alpha=0.7)
                axes[1].text(len(iterations)*0.7, final_delta*2, 
                           f'最终误差: {final_delta:.2e}', fontsize=10)
        else:
            axes[1].text(0.5, 0.5, '无价值迭代历史', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('价值迭代收敛历史')
        
        plt.tight_layout()
        plt.show()
    
    def plot_episode_analysis(self, policy: np.ndarray, start_state: str, 
                             n_episodes: int = 20, max_steps: int = 15,
                             figsize: Tuple[int, int] = (15, 10)):
        """
        绘制回合分析图
        
        参数:
            policy: 策略矩阵
            start_state: 起始状态
            n_episodes: 模拟回合数
            max_steps: 每回合最大步数
            figsize: 图形大小
        """
        print(f"\n🎮 模拟 {n_episodes} 个回合，每回合最多 {max_steps} 步")
        
        # 进行多次模拟
        all_returns = []
        all_lengths = []
        episode_data = []
        
        for episode in range(n_episodes):
            states_seq, actions_seq, rewards_seq = self.mdp.simulate_episode(
                policy, start_state, max_steps)
            
            # 计算折扣回报
            discounted_return = sum(reward * (self.mdp.gamma ** t) 
                                  for t, reward in enumerate(rewards_seq))
            
            all_returns.append(discounted_return)
            all_lengths.append(len(rewards_seq))
            episode_data.append((states_seq, actions_seq, rewards_seq, discounted_return))
        
        # 创建子图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
        
        # 1. 状态-动作轨迹图
        ax1 = fig.add_subplot(gs[0, :])
        for i, (states_seq, actions_seq, rewards_seq, ret) in enumerate(episode_data[:8]):
            steps = range(len(states_seq)-1)  # 动作序列比状态序列少1
            state_indices = [self.mdp.state_to_index[s] for s in states_seq[:-1]]
            
            # 绘制状态轨迹
            ax1.plot(steps, state_indices, 'o-', alpha=0.7, 
                    label=f'回合{i+1} (G={ret:.2f})')
            
            # 标注动作
            for j, (step, action) in enumerate(zip(steps, actions_seq)):
                if j % 2 == 0:  # 只标注部分动作以避免拥挤
                    ax1.annotate(action, (step, state_indices[j]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('状态')
        ax1.set_title(f'前8个回合的状态-动作轨迹 (从 "{start_state}" 开始)')
        ax1.set_yticks(range(self.mdp.n_states))
        ax1.set_yticklabels(self.mdp.states)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 折扣回报分布
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(all_returns, bins=min(10, n_episodes//2), alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_returns), color='red', linestyle='--', 
                   label=f'均值: {np.mean(all_returns):.3f}')
        ax2.set_xlabel('折扣回报')
        ax2.set_ylabel('频次')
        ax2.set_title('折扣回报分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回合长度分布
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(all_lengths, bins=min(10, max(all_lengths)-min(all_lengths)+1), 
                alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(all_lengths), color='red', linestyle='--', 
                   label=f'均值: {np.mean(all_lengths):.1f}')
        ax3.set_xlabel('回合长度')
        ax3.set_ylabel('频次')
        ax3.set_title('回合长度分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计摘要
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # 计算理论价值
        theoretical_value = None
        if self.mdp.current_value_function is not None:
            start_idx = self.mdp.state_to_index[start_state]
            theoretical_value = self.mdp.current_value_function[start_idx]
        
        theoretical_value_str = f"{theoretical_value:.6f}" if theoretical_value is not None else "未计算"
        simulation_error_str = f"{abs(np.mean(all_returns) - theoretical_value):.6f}" if theoretical_value is not None else "N/A"
        
        stats_text = f"""
        回合模拟统计摘要 (起始状态: {start_state})
        ═══════════════════════════════════════════════════════════════
        
        折扣回报统计:                          回合长度统计:
        • 平均值: {np.mean(all_returns):8.4f}              • 平均长度: {np.mean(all_lengths):6.2f} 步
        • 标准差: {np.std(all_returns):8.4f}              • 最短回合: {np.min(all_lengths):6d} 步
        • 最小值: {np.min(all_returns):8.4f}              • 最长回合: {np.max(all_lengths):6d} 步
        • 最大值: {np.max(all_returns):8.4f}
        
        理论对比:
        • 理论价值 V({start_state}): {theoretical_value_str}
        • 模拟误差: {simulation_error_str}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return all_returns, all_lengths


def create_grid_world_mdp() -> MarkovDecisionProcess:
    """
    创建网格世界MDP示例
    
    这是强化学习中的经典示例，智能体在4x4网格中寻找宝藏并避开陷阱。
    """
    print("\n" + "="*80)
    print("🗺️  创建示例：网格世界马尔科夫决策过程")
    print("="*80)
    
    # 定义状态（4x4网格）
    states = []
    for i in range(4):
        for j in range(4):
            states.append(f"({i},{j})")
    
    # 定义动作
    actions = ['上', '下', '左', '右']
    
    # 状态描述
    state_descriptions = {
        '(0,0)': '起始位置',
        '(0,3)': '宝藏位置 (+10奖励)',
        '(1,3)': '陷阱位置 (-10奖励)',
        '(3,3)': '终点位置'
    }
    
    # 动作描述
    action_descriptions = {
        '上': '向上移动一格',
        '下': '向下移动一格',
        '左': '向左移动一格',
        '右': '向右移动一格'
    }
    
    # 构建转移概率
    transition_probs = {}
    
    def get_next_position(pos, action):
        """根据当前位置和动作计算下一个位置"""
        i, j = eval(pos)
        if action == '上' and i > 0:
            return f"({i-1},{j})"
        elif action == '下' and i < 3:
            return f"({i+1},{j})"
        elif action == '左' and j > 0:
            return f"({i},{j-1})"
        elif action == '右' and j < 3:
            return f"({i},{j+1})"
        else:
            return pos  # 撞墙，保持原位置
    
    # 为每个状态-动作对定义转移概率
    for state in states:
        for action in actions:
            next_state = get_next_position(state, action)
            
            # 确定性转移（90%概率到达预期位置，10%概率保持原位置）
            transition_probs[(state, action, next_state)] = 0.9
            if next_state != state:
                transition_probs[(state, action, state)] = 0.1
            else:
                transition_probs[(state, action, state)] = 1.0
    
    # 定义奖励函数
    rewards = {}
    for state in states:
        for action in actions:
            if state == '(0,3)':  # 宝藏
                rewards[(state, action)] = 10.0
            elif state == '(1,3)':  # 陷阱
                rewards[(state, action)] = -10.0
            elif state == '(3,3)':  # 终点
                rewards[(state, action)] = 5.0
            else:
                rewards[(state, action)] = -0.1  # 每步小惩罚，鼓励快速到达目标
    
    # 折扣因子
    gamma = 0.9
    
    print("网格世界设置:")
    print("  • 4x4网格，16个状态")
    print("  • 4个动作：上、下、左、右")
    print("  • 特殊位置：")
    print("    - (0,3): 宝藏 (+10奖励)")
    print("    - (1,3): 陷阱 (-10奖励)")
    print("    - (3,3): 终点 (+5奖励)")
    print("  • 每步移动 -0.1 奖励（鼓励效率）")
    print("  • 90%概率成功移动，10%概率原地不动")
    
    # 创建MDP
    mdp = MarkovDecisionProcess(states, actions, transition_probs, rewards, gamma,
                               state_descriptions, action_descriptions)
    
    return mdp


def demonstrate_mdp_concepts():
    """演示马尔科夫决策过程的核心概念"""
    print("\n" + "🎯" * 40)
    print("马尔科夫决策过程 (MDP) 核心概念演示")
    print("🎯" * 40)
    
    # 创建示例MDP
    mdp = create_grid_world_mdp()
    
    # 打印详细信息
    mdp.print_detailed_summary()
    
    # 创建可视化器
    visualizer = MDPVisualizer(mdp)
    
    print("\n" + "="*60)
    print("🔄 第一步：策略迭代算法")
    print("="*60)
    
    # 策略迭代
    optimal_policy_pi, optimal_value_pi = mdp.policy_iteration(verbose=True)
    
    print("\n" + "="*60)
    print("⚡ 第二步：价值迭代算法")
    print("="*60)
    
    # 价值迭代
    optimal_policy_vi, optimal_value_vi = mdp.value_iteration(verbose=True)
    
    print("\n" + "="*60)
    print("📊 第三步：计算Q函数")
    print("="*60)
    
    # 计算Q函数
    Q_function = mdp.compute_q_function(optimal_policy_pi)
    print("动作价值函数 Q(s,a) 计算完成")
    
    print("\n" + "="*60)
    print("📈 第四步：生成可视化分析")
    print("="*60)
    
    print("正在生成可视化图表...")
    
    # 1. 策略可视化
    print("1. 最优策略可视化")
    visualizer.plot_policy_visualization(optimal_policy_pi, "策略迭代 - 最优策略")
    
    # 2. 价值函数可视化
    print("2. 状态价值函数可视化")
    visualizer.plot_value_function(optimal_value_pi, "策略迭代 - 最优价值函数")
    
    # 3. Q函数可视化
    print("3. 动作价值函数可视化")
    visualizer.plot_q_function(Q_function, "动作价值函数 Q(s,a)")
    
    # 4. 收敛历史
    print("4. 算法收敛历史")
    visualizer.plot_convergence_history()
    
    # 5. 回合模拟
    print("5. 回合模拟分析")
    returns, lengths = visualizer.plot_episode_analysis(
        optimal_policy_pi, '(0,0)', n_episodes=25, max_steps=20)
    
    # 算法比较
    print("\n" + "="*80)
    print("🔍 算法比较分析")
    print("="*80)
    
    print("策略迭代 vs 价值迭代:")
    print(f"  策略迭代轮数: {len(mdp.policy_iteration_history)}")
    print(f"  价值迭代轮数: {len(mdp.value_iteration_history)}")
    
    # 比较最优价值函数
    value_diff = np.max(np.abs(optimal_value_pi - optimal_value_vi))
    print(f"  价值函数差异: {value_diff:.8f}")
    
    # 比较最优策略
    policy_diff = np.max(np.abs(optimal_policy_pi - optimal_policy_vi))
    print(f"  策略差异: {policy_diff:.8f}")
    
    # 最终总结
    print("\n" + "="*80)
    print("📋 马尔科夫决策过程分析总结")
    print("="*80)
    
    print(f"\n🎯 关键发现:")
    best_state_idx = np.argmax(optimal_value_pi)
    worst_state_idx = np.argmin(optimal_value_pi)
    print(f"  • 最优状态: {mdp.states[best_state_idx]} (V* = {optimal_value_pi[best_state_idx]:.3f})")
    print(f"  • 最差状态: {mdp.states[worst_state_idx]} (V* = {optimal_value_vi[worst_state_idx]:.3f})")
    
    print(f"\n📊 模拟验证:")
    print(f"  • 平均折扣回报: {np.mean(returns):.3f}")
    print(f"  • 回报标准差: {np.std(returns):.3f}")
    print(f"  • 平均回合长度: {np.mean(lengths):.1f} 步")
    
    print(f"\n💡 教学要点:")
    print(f"  1. MDP扩展了MRP，增加了动作选择")
    print(f"  2. 策略定义了在每个状态下的行为")
    print(f"  3. 贝尔曼最优方程描述了最优价值函数")
    print(f"  4. 策略迭代和价值迭代都能找到最优策略")
    print(f"  5. Q函数帮助理解状态-动作对的价值")


if __name__ == "__main__":
    # 运行完整的MDP概念演示
    demonstrate_mdp_concepts()