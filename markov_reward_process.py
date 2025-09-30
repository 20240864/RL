"""
马尔科夫奖励过程 (Markov Reward Process, MRP) 教学代码

本代码详细实现了马尔科夫奖励过程的核心概念，包括：
1. 状态空间和转移概率矩阵
2. 奖励函数
3. 价值函数的计算（解析解和迭代解）
4. 可视化功能

作者：教学示例代码
用途：强化学习课堂教学
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class MarkovRewardProcess:
    """
    马尔科夫奖励过程类
    
    马尔科夫奖励过程是一个四元组 (S, P, R, γ)，其中：
    - S: 状态空间 (State Space)
    - P: 状态转移概率矩阵 (Transition Probability Matrix)
    - R: 奖励函数 (Reward Function)
    - γ: 折扣因子 (Discount Factor)
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray, 
                 rewards: np.ndarray, gamma: float = 0.9):
        """
        初始化马尔科夫奖励过程
        
        参数:
            states: 状态名称列表
            transition_matrix: 状态转移概率矩阵 P[i,j] = P(s_{t+1}=j | s_t=i)
            rewards: 奖励向量 R[i] = E[R_{t+1} | s_t=i]
            gamma: 折扣因子，范围 [0, 1]
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_index = {state: i for i, state in enumerate(states)}
        
        # 验证输入参数
        self._validate_inputs(transition_matrix, rewards, gamma)
        
        self.P = transition_matrix
        self.R = rewards
        self.gamma = gamma
        
        # 计算价值函数
        self.V_analytical = None
        self.V_iterative = None
        self.convergence_history = []
        
    def _validate_inputs(self, P: np.ndarray, R: np.ndarray, gamma: float):
        """验证输入参数的有效性"""
        # 检查转移概率矩阵
        if P.shape != (self.n_states, self.n_states):
            raise ValueError(f"转移概率矩阵形状应为 ({self.n_states}, {self.n_states})")
        
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError("转移概率矩阵每行和应为1")
        
        if np.any(P < 0):
            raise ValueError("转移概率不能为负数")
        
        # 检查奖励向量
        if R.shape != (self.n_states,):
            raise ValueError(f"奖励向量长度应为 {self.n_states}")
        
        # 检查折扣因子
        if not 0 <= gamma <= 1:
            raise ValueError("折扣因子应在 [0, 1] 范围内")
    
    def compute_value_function_analytical(self) -> np.ndarray:
        """
        使用解析解计算价值函数
        
        价值函数的解析解：V = (I - γP)^(-1) * R
        其中 I 是单位矩阵
        
        返回:
            价值函数向量
        """
        try:
            # V = (I - γP)^(-1) * R
            I = np.eye(self.n_states)
            matrix_to_invert = I - self.gamma * self.P
            
            # 检查矩阵是否可逆
            if np.linalg.det(matrix_to_invert) == 0:
                raise np.linalg.LinAlgError("矩阵 (I - γP) 不可逆")
            
            self.V_analytical = np.linalg.solve(matrix_to_invert, self.R)
            return self.V_analytical
            
        except np.linalg.LinAlgError as e:
            print(f"解析解计算失败: {e}")
            return None
    
    def compute_value_function_iterative(self, max_iterations: int = 1000, 
                                       tolerance: float = 1e-6) -> np.ndarray:
        """
        使用价值迭代算法计算价值函数
        
        迭代公式：V_{k+1}(s) = R(s) + γ * Σ P(s'|s) * V_k(s')
        
        参数:
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        返回:
            价值函数向量
        """
        # 初始化价值函数
        V = np.zeros(self.n_states)
        self.convergence_history = []
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            # 价值迭代更新
            V = self.R + self.gamma * np.dot(self.P, V)
            
            # 记录收敛历史
            max_change = np.max(np.abs(V - V_old))
            self.convergence_history.append(max_change)
            
            # 检查收敛
            if max_change < tolerance:
                print(f"价值迭代在第 {iteration + 1} 次迭代后收敛")
                break
        else:
            print(f"价值迭代在 {max_iterations} 次迭代后未收敛")
        
        self.V_iterative = V
        return V
    
    def get_state_value(self, state: str, method: str = 'analytical') -> float:
        """
        获取特定状态的价值
        
        参数:
            state: 状态名称
            method: 计算方法 ('analytical' 或 'iterative')
            
        返回:
            状态价值
        """
        if state not in self.state_to_index:
            raise ValueError(f"状态 '{state}' 不存在")
        
        index = self.state_to_index[state]
        
        if method == 'analytical':
            if self.V_analytical is None:
                self.compute_value_function_analytical()
            return self.V_analytical[index]
        elif method == 'iterative':
            if self.V_iterative is None:
                self.compute_value_function_iterative()
            return self.V_iterative[index]
        else:
            raise ValueError("方法必须是 'analytical' 或 'iterative'")
    
    def simulate_episode(self, start_state: str, max_steps: int = 100) -> Tuple[List[str], List[float]]:
        """
        模拟一个回合
        
        参数:
            start_state: 起始状态
            max_steps: 最大步数
            
        返回:
            状态序列和奖励序列
        """
        if start_state not in self.state_to_index:
            raise ValueError(f"起始状态 '{start_state}' 不存在")
        
        states_sequence = [start_state]
        rewards_sequence = []
        
        current_state_idx = self.state_to_index[start_state]
        
        for _ in range(max_steps):
            # 获取当前状态的奖励
            reward = self.R[current_state_idx]
            rewards_sequence.append(reward)
            
            # 根据转移概率选择下一个状态
            next_state_idx = np.random.choice(
                self.n_states, 
                p=self.P[current_state_idx]
            )
            
            next_state = self.states[next_state_idx]
            states_sequence.append(next_state)
            current_state_idx = next_state_idx
        
        return states_sequence, rewards_sequence
    
    def print_summary(self):
        """打印MRP的详细信息"""
        print("=" * 60)


class MRPVisualizer:
    """马尔科夫奖励过程可视化类"""
    
    def __init__(self, mrp: MarkovRewardProcess):
        """
        初始化可视化器
        
        参数:
            mrp: 马尔科夫奖励过程实例
        """
        self.mrp = mrp
        
    def plot_state_transition_graph(self, figsize: Tuple[int, int] = (12, 8), 
                                  threshold: float = 0.01):
        """
        绘制状态转移图
        
        参数:
            figsize: 图形大小
            threshold: 显示转移概率的最小阈值
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for state in self.mrp.states:
            G.add_node(state)
        
        # 添加边（转移概率大于阈值的）
        edge_labels = {}
        for i, from_state in enumerate(self.mrp.states):
            for j, to_state in enumerate(self.mrp.states):
                prob = self.mrp.P[i, j]
                if prob > threshold:
                    G.add_edge(from_state, to_state, weight=prob)
                    edge_labels[(from_state, to_state)] = f'{prob:.3f}'
        
        # 设置布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for state in G.nodes():
            idx = self.mrp.state_to_index[state]
            reward = self.mrp.R[idx]
            
            # 根据奖励值设置颜色
            if reward > 0:
                node_colors.append('lightgreen')
            elif reward < 0:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
            
            # 根据奖励绝对值设置大小
            node_sizes.append(1000 + abs(reward) * 200)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, ax=ax)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              arrowstyle='->', ax=ax)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        # 绘制边标签（转移概率）
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)
        
        # 添加奖励信息
        for state, (x, y) in pos.items():
            idx = self.mrp.state_to_index[state]
            reward = self.mrp.R[idx]
            ax.text(x, y-0.15, f'R={reward:.2f}', 
                   horizontalalignment='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title('马尔科夫奖励过程 - 状态转移图', fontsize=16, fontweight='bold')
        ax.text(0.02, 0.98, f'折扣因子 γ = {self.mrp.gamma}', 
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=10, label='正奖励状态'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='负奖励状态'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='零奖励状态')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_transition_matrix_heatmap(self, figsize: Tuple[int, int] = (10, 8)):
        """绘制转移概率矩阵热力图"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建热力图
        im = ax.imshow(self.mrp.P, cmap='Blues', aspect='auto')
        
        # 设置坐标轴标签
        ax.set_xticks(range(self.mrp.n_states))
        ax.set_yticks(range(self.mrp.n_states))
        ax.set_xticklabels(self.mrp.states)
        ax.set_yticklabels(self.mrp.states)
        
        # 添加数值标注
        for i in range(self.mrp.n_states):
            for j in range(self.mrp.n_states):
                text = ax.text(j, i, f'{self.mrp.P[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # 设置标题和标签
        ax.set_title('状态转移概率矩阵', fontsize=16, fontweight='bold')
        ax.set_xlabel('目标状态', fontsize=12)
        ax.set_ylabel('起始状态', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('转移概率', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_value_function_comparison(self, figsize: Tuple[int, int] = (12, 6)):
        """比较解析解和迭代解的价值函数"""
        if self.mrp.V_analytical is None:
            self.mrp.compute_value_function_analytical()
        if self.mrp.V_iterative is None:
            self.mrp.compute_value_function_iterative()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：价值函数比较
        x = range(self.mrp.n_states)
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], self.mrp.V_analytical, width, 
               label='解析解', alpha=0.8, color='skyblue')
        ax1.bar([i + width/2 for i in x], self.mrp.V_iterative, width, 
               label='迭代解', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('状态')
        ax1.set_ylabel('价值函数值')
        ax1.set_title('价值函数：解析解 vs 迭代解')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.mrp.states)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：收敛历史
        if self.mrp.convergence_history:
            ax2.plot(self.mrp.convergence_history, 'b-', linewidth=2, marker='o')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('最大变化量')
            ax2.set_title('价值迭代收敛历史')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_reward_function(self, figsize: Tuple[int, int] = (10, 6)):
        """绘制奖励函数"""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red' if r < 0 else 'green' if r > 0 else 'blue' for r in self.mrp.R]
        bars = ax.bar(self.mrp.states, self.mrp.R, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar, reward in zip(bars, self.mrp.R):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                   f'{reward:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        ax.set_xlabel('状态')
        ax.set_ylabel('奖励值')
        ax.set_title('奖励函数 R(s)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_episode_simulation(self, start_state: str, n_episodes: int = 5, 
                               max_steps: int = 20, figsize: Tuple[int, int] = (14, 8)):
        """可视化回合模拟"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        all_returns = []
        all_episode_lengths = []
        
        # 模拟多个回合
        for episode in range(n_episodes):
            states_seq, rewards_seq = self.mrp.simulate_episode(start_state, max_steps)
            
            # 计算折扣回报
            discounted_return = sum(reward * (self.mrp.gamma ** t) 
                                  for t, reward in enumerate(rewards_seq))
            all_returns.append(discounted_return)
            all_episode_lengths.append(len(states_seq) - 1)
            
            # 绘制前3个回合的轨迹
            if episode < 3:
                ax1.plot(range(len(rewards_seq)), rewards_seq, 
                        marker='o', label=f'回合 {episode+1}', alpha=0.7)
        
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('即时奖励')
        ax1.set_title('回合轨迹 - 即时奖励')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制回报分布
        ax2.hist(all_returns, bins=min(10, n_episodes), alpha=0.7, color='skyblue')
        ax2.set_xlabel('折扣回报')
        ax2.set_ylabel('频次')
        ax2.set_title('折扣回报分布')
        ax2.grid(True, alpha=0.3)
        
        # 绘制状态访问频率
        state_counts = {state: 0 for state in self.mrp.states}
        for _ in range(100):  # 更多样本用于统计
            states_seq, _ = self.mrp.simulate_episode(start_state, max_steps)
            for state in states_seq:
                state_counts[state] += 1
        
        states_list = list(state_counts.keys())
        counts_list = list(state_counts.values())
        ax3.bar(states_list, counts_list, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('状态')
        ax3.set_ylabel('访问次数')
        ax3.set_title('状态访问频率 (100个回合)')
        ax3.grid(True, alpha=0.3)
        
        # 绘制价值函数（如果已计算）
        if self.mrp.V_analytical is not None:
            ax4.bar(self.mrp.states, self.mrp.V_analytical, alpha=0.7, color='orange')
            ax4.set_xlabel('状态')
            ax4.set_ylabel('状态价值')
            ax4.set_title('理论状态价值函数')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return all_returns, all_episode_lengths
        print("马尔科夫奖励过程 (Markov Reward Process) 摘要")
        print("=" * 60)
        
        print(f"状态空间 S: {self.states}")
        print(f"状态数量: {self.n_states}")
        print(f"折扣因子 γ: {self.gamma}")
        
        print("\n状态转移概率矩阵 P:")
        print("-" * 40)
        # 创建格式化的转移矩阵表格
        header = "从\\到".ljust(8) + "".join([f"{s:>8}" for s in self.states])
        print(header)
        for i, from_state in enumerate(self.states):
            row = f"{from_state:<8}" + "".join([f"{self.P[i,j]:>8.3f}" for j in range(self.n_states)])
            print(row)
        
        print(f"\n奖励函数 R:")
        print("-" * 20)
        for i, state in enumerate(self.states):
            print(f"R({state}) = {self.R[i]:>6.2f}")
        
        # 如果已计算价值函数，则显示
        if self.V_analytical is not None:
            print(f"\n价值函数 V (解析解):")
            print("-" * 25)
            for i, state in enumerate(self.states):
                print(f"V({state}) = {self.V_analytical[i]:>6.3f}")
        
        if self.V_iterative is not None:
            print(f"\n价值函数 V (迭代解):")
            print("-" * 25)
            for i, state in enumerate(self.states):
                print(f"V({state}) = {self.V_iterative[i]:>6.3f}")
        
        print("=" * 60)