"""
马尔可夫奖励过程 (Markov Reward Process, MRP) 详细实现

本文件提供了MRP的完整实现，包括：
1. 核心数学理论的严格实现
2. 多种价值函数计算方法
3. 丰富的可视化功能
4. 详细的教学示例和分析

作者：强化学习教学代码
版本：2.0
用途：深入理解马尔可夫奖励过程的数学原理和实际应用
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MarkovRewardProcessDetailed:
    """
    马尔可夫奖励过程详细实现类
    
    马尔可夫奖励过程是一个四元组 (S, P, R, γ)：
    - S: 状态空间 (State Space)
    - P: 状态转移概率矩阵 (Transition Probability Matrix)
    - R: 奖励函数 (Reward Function)
    - γ: 折扣因子 (Discount Factor)
    
    核心特性：
    1. 马尔可夫性质：P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)
    2. 价值函数：V(s) = E[G_t | S_t = s]，其中 G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}
    3. 贝尔曼方程：V(s) = R(s) + γ * Σ_{s'} P(s'|s) * V(s')
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray, 
                 rewards: np.ndarray, gamma: float, state_descriptions: Optional[Dict[str, str]] = None):
        """
        初始化马尔可夫奖励过程
        
        参数:
            states: 状态名称列表
            transition_matrix: 状态转移概率矩阵 P[i,j] = P(S_{t+1}=j | S_t=i)
            rewards: 奖励函数 R[i] = E[R_{t+1} | S_t=i]
            gamma: 折扣因子 γ ∈ [0,1]
            state_descriptions: 状态描述字典（可选）
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_index = {state: i for i, state in enumerate(states)}
        self.transition_matrix = np.array(transition_matrix)
        self.rewards = np.array(rewards)
        self.gamma = gamma
        self.state_descriptions = state_descriptions or {}
        
        # 验证输入参数
        self._validate_parameters()
        
        # 存储计算历史
        self.convergence_history = []
        self.analytical_solution = None
        self.iterative_solution = None
        
        print(f"✅ 马尔可夫奖励过程初始化完成")
        print(f"   状态数量: {self.n_states}")
        print(f"   折扣因子: {self.gamma}")
        print(f"   状态列表: {self.states}")
    
    def _validate_parameters(self):
        """验证输入参数的有效性"""
        # 检查转移概率矩阵
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"转移概率矩阵形状错误: 期望 {(self.n_states, self.n_states)}, 实际 {self.transition_matrix.shape}")
        
        # 检查概率矩阵每行和为1
        row_sums = np.sum(self.transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-10):
            raise ValueError(f"转移概率矩阵每行和必须为1, 实际: {row_sums}")
        
        # 检查概率非负
        if np.any(self.transition_matrix < 0):
            raise ValueError("转移概率不能为负数")
        
        # 检查奖励函数
        if len(self.rewards) != self.n_states:
            raise ValueError(f"奖励函数长度错误: 期望 {self.n_states}, 实际 {len(self.rewards)}")
        
        # 检查折扣因子
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"折扣因子必须在[0,1]范围内, 实际: {self.gamma}")
    
    def compute_value_function_analytical(self) -> np.ndarray:
        """
        使用解析解计算价值函数
        
        贝尔曼方程的矩阵形式：V = R + γPV
        重新整理得到：V = (I - γP)^(-1) * R
        
        返回:
            价值函数向量 V
        """
        print("\n🧮 计算价值函数 - 解析解方法")
        print("=" * 50)
        
        try:
            # 计算 (I - γP)
            I = np.eye(self.n_states)
            matrix_to_invert = I - self.gamma * self.transition_matrix
            
            print(f"计算矩阵 (I - γP):")
            print(f"I (单位矩阵):\n{I}")
            print(f"γP (折扣转移矩阵):\n{self.gamma * self.transition_matrix}")
            print(f"(I - γP):\n{matrix_to_invert}")
            
            # 检查矩阵是否可逆
            det = np.linalg.det(matrix_to_invert)
            print(f"矩阵行列式: {det:.6f}")
            
            if abs(det) < 1e-10:
                raise np.linalg.LinAlgError("矩阵接近奇异，无法求逆")
            
            # 计算逆矩阵
            inverse_matrix = np.linalg.inv(matrix_to_invert)
            print(f"逆矩阵 (I - γP)^(-1):\n{inverse_matrix}")
            
            # 计算价值函数
            self.analytical_solution = inverse_matrix @ self.rewards
            
            print(f"\n✅ 解析解计算完成")
            print("价值函数结果:")
            for i, (state, value) in enumerate(zip(self.states, self.analytical_solution)):
                print(f"  V({state}) = {value:.6f}")
            
            return self.analytical_solution
            
        except np.linalg.LinAlgError as e:
            print(f"❌ 解析解计算失败: {e}")
            print("可能原因: γ = 1 且存在吸收状态，或矩阵奇异")
            return None
    
    def compute_value_function_iterative(self, max_iterations: int = 1000, 
                                       tolerance: float = 1e-8, verbose: bool = True) -> np.ndarray:
        """
        使用价值迭代算法计算价值函数
        
        迭代公式：V_{k+1}(s) = R(s) + γ * Σ_{s'} P(s'|s) * V_k(s')
        
        参数:
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            verbose: 是否显示详细信息
            
        返回:
            价值函数向量 V
        """
        if verbose:
            print("\n🔄 计算价值函数 - 价值迭代方法")
            print("=" * 50)
            print(f"最大迭代次数: {max_iterations}")
            print(f"收敛容忍度: {tolerance}")
        
        # 初始化价值函数
        V = np.zeros(self.n_states)
        self.convergence_history = []
        
        for iteration in range(max_iterations):
            # 保存上一次的价值函数
            V_old = V.copy()
            
            # 价值迭代更新
            V_new = self.rewards + self.gamma * (self.transition_matrix @ V)
            
            # 计算变化量
            delta = np.max(np.abs(V_new - V))
            self.convergence_history.append(delta)
            
            if verbose and (iteration < 10 or iteration % 50 == 0):
                print(f"迭代 {iteration:3d}: 最大变化量 = {delta:.8f}")
                if iteration < 5:  # 显示前几次迭代的详细信息
                    print(f"  V = {V_new}")
            
            # 检查收敛
            if delta < tolerance:
                if verbose:
                    print(f"✅ 在第 {iteration} 次迭代后收敛")
                    print(f"最终收敛精度: {delta:.10f}")
                break
            
            V = V_new
        else:
            if verbose:
                print(f"⚠️  达到最大迭代次数 {max_iterations}，未完全收敛")
                print(f"最终变化量: {delta:.8f}")
        
        self.iterative_solution = V
        
        if verbose:
            print("\n价值函数结果:")
            for i, (state, value) in enumerate(zip(self.states, V)):
                print(f"  V({state}) = {value:.6f}")
        
        return V
    
    def simulate_episode(self, start_state: str, max_steps: int = 100, 
                        random_seed: Optional[int] = None) -> Tuple[List[str], List[float]]:
        """
        模拟一个回合的状态转移和奖励序列
        
        参数:
            start_state: 起始状态
            max_steps: 最大步数
            random_seed: 随机种子
            
        返回:
            (状态序列, 奖励序列)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if start_state not in self.state_to_index:
            raise ValueError(f"起始状态 '{start_state}' 不在状态空间中")
        
        states_sequence = [start_state]
        rewards_sequence = []
        current_state_idx = self.state_to_index[start_state]
        
        for step in range(max_steps):
            # 获取当前状态的奖励
            reward = self.rewards[current_state_idx]
            rewards_sequence.append(reward)
            
            # 根据转移概率选择下一个状态
            transition_probs = self.transition_matrix[current_state_idx]
            next_state_idx = np.random.choice(self.n_states, p=transition_probs)
            next_state = self.states[next_state_idx]
            
            states_sequence.append(next_state)
            current_state_idx = next_state_idx
        
        return states_sequence, rewards_sequence
    
    def compute_discounted_return(self, rewards_sequence: List[float]) -> float:
        """
        计算折扣回报 G_t = Σ_{k=0}^∞ γ^k * R_{t+k+1}
        
        参数:
            rewards_sequence: 奖励序列
            
        返回:
            折扣回报
        """
        return sum(reward * (self.gamma ** t) for t, reward in enumerate(rewards_sequence))
    
    def analyze_stationary_distribution(self) -> np.ndarray:
        """
        分析马尔可夫链的平稳分布
        
        平稳分布 π 满足：π = πP，即 π(I - P) = 0
        
        返回:
            平稳分布向量
        """
        print("\n📊 分析马尔可夫链的平稳分布")
        print("=" * 50)
        
        try:
            # 计算转移矩阵的特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            
            # 找到特征值为1的特征向量（平稳分布）
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_vector = np.real(eigenvectors[:, stationary_idx])
            
            # 归一化使其成为概率分布
            stationary_distribution = stationary_vector / np.sum(stationary_vector)
            
            # 确保非负
            if np.any(stationary_distribution < 0):
                stationary_distribution = np.abs(stationary_distribution)
                stationary_distribution = stationary_distribution / np.sum(stationary_distribution)
            
            print("平稳分布结果:")
            for state, prob in zip(self.states, stationary_distribution):
                print(f"  π({state}) = {prob:.6f}")
            
            # 验证平稳分布
            verification = stationary_distribution @ self.transition_matrix
            error = np.max(np.abs(verification - stationary_distribution))
            print(f"\n验证误差: {error:.10f}")
            
            return stationary_distribution
            
        except Exception as e:
            print(f"❌ 平稳分布计算失败: {e}")
            return None
    
    def print_detailed_summary(self):
        """打印详细的MRP信息摘要"""
        print("\n" + "=" * 80)
        print("马尔可夫奖励过程 (MRP) 详细信息摘要")
        print("=" * 80)
        
        print(f"\n📋 基本信息:")
        print(f"   状态空间大小: {self.n_states}")
        print(f"   状态列表: {self.states}")
        print(f"   折扣因子 γ: {self.gamma}")
        
        if self.state_descriptions:
            print(f"\n📝 状态描述:")
            for state, desc in self.state_descriptions.items():
                print(f"   {state}: {desc}")
        
        print(f"\n🎯 奖励函数:")
        for i, (state, reward) in enumerate(zip(self.states, self.rewards)):
            print(f"   R({state}) = {reward:8.3f}")
        
        print(f"\n🔄 状态转移概率矩阵:")
        print("     " + "".join(f"{state:>8}" for state in self.states))
        for i, from_state in enumerate(self.states):
            row_str = f"{from_state:>4} "
            for j in range(self.n_states):
                row_str += f"{self.transition_matrix[i, j]:8.3f}"
            print(row_str)
        
        # 显示价值函数结果
        if self.analytical_solution is not None or self.iterative_solution is not None:
            print(f"\n💰 价值函数结果:")
            if self.analytical_solution is not None:
                print("   解析解:")
                for state, value in zip(self.states, self.analytical_solution):
                    print(f"     V({state}) = {value:10.6f}")
            
            if self.iterative_solution is not None:
                print("   迭代解:")
                for state, value in zip(self.states, self.iterative_solution):
                    print(f"     V({state}) = {value:10.6f}")
                
                if self.analytical_solution is not None:
                    print("   差异分析:")
                    for state, v_ana, v_iter in zip(self.states, self.analytical_solution, self.iterative_solution):
                        diff = abs(v_ana - v_iter)
                        print(f"     |V_ana({state}) - V_iter({state})| = {diff:.8f}")


class MRPVisualizer:
    """马尔可夫奖励过程可视化类"""
    
    def __init__(self, mrp: MarkovRewardProcessDetailed):
        """
        初始化可视化器
        
        参数:
            mrp: 马尔可夫奖励过程实例
        """
        self.mrp = mrp
        plt.style.use('default')  # 使用默认样式
        # 重新设置中文字体，确保在样式设置后生效
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_state_transition_graph(self, figsize: Tuple[int, int] = (12, 8), 
                                  min_edge_weight: float = 0.01):
        """
        绘制状态转移图
        
        参数:
            figsize: 图形大小
            min_edge_weight: 最小边权重阈值
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for i, state in enumerate(self.mrp.states):
            reward = self.mrp.rewards[i]
            G.add_node(state, reward=reward)
        
        # 添加边（只显示概率大于阈值的转移）
        for i, from_state in enumerate(self.mrp.states):
            for j, to_state in enumerate(self.mrp.states):
                prob = self.mrp.transition_matrix[i, j]
                if prob > min_edge_weight:
                    G.add_edge(from_state, to_state, weight=prob)
        
        # 设置布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for state in G.nodes():
            reward = G.nodes[state]['reward']
            if reward > 0:
                node_colors.append('lightgreen')
            elif reward < 0:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
            node_sizes.append(max(500, abs(reward) * 50 + 300))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*3 for w in weights],
                              alpha=0.6, edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->')
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 添加边标签（转移概率）
        edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        # 添加奖励信息
        reward_labels = {}
        for state in G.nodes():
            reward = G.nodes[state]['reward']
            reward_labels[state] = f'R={reward:.1f}'
        
        # 调整标签位置
        pos_rewards = {node: (x, y-0.15) for node, (x, y) in pos.items()}
        for node, (x, y) in pos_rewards.items():
            ax.text(x, y, reward_labels[node], ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=10)
        
        ax.set_title('马尔可夫奖励过程 - 状态转移图\n'
                    f'(γ={self.mrp.gamma}, 只显示概率>{min_edge_weight}的转移)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
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
        
        plt.tight_layout()
        plt.show()
    
    def plot_transition_matrix_heatmap(self, figsize: Tuple[int, int] = (10, 8)):
        """绘制转移概率矩阵热力图"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建热力图
        im = ax.imshow(self.mrp.transition_matrix, cmap='Blues', aspect='auto')
        
        # 设置坐标轴标签
        ax.set_xticks(range(self.mrp.n_states))
        ax.set_yticks(range(self.mrp.n_states))
        ax.set_xticklabels(self.mrp.states)
        ax.set_yticklabels(self.mrp.states)
        
        # 添加数值标注
        for i in range(self.mrp.n_states):
            for j in range(self.mrp.n_states):
                prob = self.mrp.transition_matrix[i, j]
                color = 'white' if prob > 0.5 else 'black'
                ax.text(j, i, f'{prob:.3f}', ha='center', va='center', 
                       color=color, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('转移概率', rotation=270, labelpad=20)
        
        ax.set_title('状态转移概率矩阵热力图\nP[i,j] = P(S_{t+1}=j | S_t=i)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('目标状态 (S_{t+1})')
        ax.set_ylabel('当前状态 (S_t)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_value_function_comparison(self, figsize: Tuple[int, int] = (12, 6)):
        """绘制价值函数比较图"""
        if self.mrp.analytical_solution is None and self.mrp.iterative_solution is None:
            print("❌ 没有可用的价值函数结果进行比较")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：价值函数柱状图
        x = np.arange(len(self.mrp.states))
        width = 0.35
        
        if self.mrp.analytical_solution is not None:
            bars1 = ax1.bar(x - width/2, self.mrp.analytical_solution, width, 
                           label='解析解', alpha=0.8, color='skyblue')
        
        if self.mrp.iterative_solution is not None:
            bars2 = ax1.bar(x + width/2, self.mrp.iterative_solution, width,
                           label='迭代解', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('状态')
        ax1.set_ylabel('价值函数 V(s)')
        ax1.set_title('价值函数比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.mrp.states)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标注
        if self.mrp.analytical_solution is not None:
            for i, v in enumerate(self.mrp.analytical_solution):
                ax1.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        if self.mrp.iterative_solution is not None:
            for i, v in enumerate(self.mrp.iterative_solution):
                ax1.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        # 右图：收敛历史
        if self.mrp.convergence_history:
            ax2.semilogy(self.mrp.convergence_history, 'b-', linewidth=2)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('最大变化量 (对数尺度)')
            ax2.set_title('价值迭代收敛历史')
            ax2.grid(True, alpha=0.3)
            
            # 标注收敛点
            final_error = self.mrp.convergence_history[-1]
            ax2.axhline(y=final_error, color='red', linestyle='--', alpha=0.7)
            ax2.text(len(self.mrp.convergence_history)*0.7, final_error*2, 
                    f'最终误差: {final_error:.2e}', fontsize=10)
        else:
            ax2.text(0.5, 0.5, '无收敛历史数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('价值迭代收敛历史')
        
        plt.tight_layout()
        plt.show()
    
    def plot_reward_function(self, figsize: Tuple[int, int] = (10, 6)):
        """绘制奖励函数图"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建颜色映射
        colors = ['red' if r < 0 else 'green' if r > 0 else 'gray' for r in self.mrp.rewards]
        
        bars = ax.bar(self.mrp.states, self.mrp.rewards, color=colors, alpha=0.7)
        
        # 添加数值标注
        for bar, reward in zip(bars, self.mrp.rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                   f'{reward:.1f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_xlabel('状态')
        ax.set_ylabel('奖励值 R(s)')
        ax.set_title('马尔可夫奖励过程 - 奖励函数')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_episode_simulation(self, start_state: str, n_episodes: int = 20, 
                              max_steps: int = 15, figsize: Tuple[int, int] = (15, 10)):
        """
        绘制回合模拟分析图
        
        参数:
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
            states_seq, rewards_seq = self.mrp.simulate_episode(start_state, max_steps)
            discounted_return = self.mrp.compute_discounted_return(rewards_seq)
            
            all_returns.append(discounted_return)
            all_lengths.append(len(rewards_seq))
            episode_data.append((states_seq, rewards_seq, discounted_return))
        
        # 创建子图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
        
        # 1. 回合轨迹图
        ax1 = fig.add_subplot(gs[0, :])
        for i, (states_seq, rewards_seq, ret) in enumerate(episode_data[:10]):  # 只显示前10个回合
            steps = range(len(states_seq))
            state_indices = [self.mrp.state_to_index[s] for s in states_seq]
            ax1.plot(steps, state_indices, 'o-', alpha=0.7, label=f'回合{i+1} (G={ret:.2f})')
        
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('状态索引')
        ax1.set_title(f'前10个回合的状态轨迹 (从 "{start_state}" 开始)')
        ax1.set_yticks(range(self.mrp.n_states))
        ax1.set_yticklabels(self.mrp.states)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 折扣回报分布
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(all_returns, bins=min(10, n_episodes//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_returns), color='red', linestyle='--', label=f'均值: {np.mean(all_returns):.3f}')
        ax2.set_xlabel('折扣回报 G')
        ax2.set_ylabel('频次')
        ax2.set_title('折扣回报分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 回合长度分布
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(all_lengths, bins=min(10, max(all_lengths)-min(all_lengths)+1), 
                alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(all_lengths), color='red', linestyle='--', label=f'均值: {np.mean(all_lengths):.1f}')
        ax3.set_xlabel('回合长度')
        ax3.set_ylabel('频次')
        ax3.set_title('回合长度分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计摘要
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # 计算理论价值（如果可用）
        theoretical_value = None
        if self.mrp.analytical_solution is not None:
            start_idx = self.mrp.state_to_index[start_state]
            theoretical_value = self.mrp.analytical_solution[start_idx]
        elif self.mrp.iterative_solution is not None:
            start_idx = self.mrp.state_to_index[start_state]
            theoretical_value = self.mrp.iterative_solution[start_idx]
        
        stats_text = f"""
        模拟统计摘要 (起始状态: {start_state})
        ═══════════════════════════════════════════════════════════════
        
        折扣回报统计:
        • 平均值: {np.mean(all_returns):.6f}
        • 标准差: {np.std(all_returns):.6f}
        • 最小值: {np.min(all_returns):.6f}
        • 最大值: {np.max(all_returns):.6f}
        
        回合长度统计:
        • 平均长度: {np.mean(all_lengths):.2f} 步
        • 最短回合: {np.min(all_lengths)} 步
        • 最长回合: {np.max(all_lengths)} 步
        
        理论对比:
        • 理论价值 V({start_state}): {f"{theoretical_value:.6f}" if theoretical_value is not None else "未计算"}
        • 模拟误差: {f"{abs(np.mean(all_returns) - theoretical_value):.6f}" if theoretical_value is not None else "N/A"}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='sans-serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return all_returns, all_lengths


def create_student_learning_mrp() -> MarkovRewardProcessDetailed:
    """
    创建学生学习过程的详细MRP示例
    
    这是强化学习教学中的经典示例，展示了学生在不同学习状态间的转移。
    """
    print("\n" + "="*80)
    print("📚 创建示例：学生学习过程马尔可夫奖励过程")
    print("="*80)
    
    # 定义状态和描述
    states = ['专心学习', '分心', '睡觉', '考试']
    state_descriptions = {
        '专心学习': '学生全神贯注地学习，吸收知识效率高',
        '分心': '学生注意力不集中，学习效果差，容易受干扰',
        '睡觉': '学生休息状态，既不学习也不娱乐',
        '考试': '学生参加考试，根据之前的学习状态获得相应结果'
    }
    
    # 定义转移概率矩阵
    # 专心学习 -> [专心学习, 分心, 睡觉, 考试]
    # 分心 -> [专心学习, 分心, 睡觉, 考试]
    # 睡觉 -> [专心学习, 分心, 睡觉, 考试]
    # 考试 -> [专心学习, 分心, 睡觉, 考试] (考试后重新开始学习周期)
    transition_matrix = np.array([
        [0.6, 0.2, 0.1, 0.1],  # 从专心学习
        [0.3, 0.3, 0.3, 0.1],  # 从分心
        [0.2, 0.1, 0.6, 0.1],  # 从睡觉
        [0.4, 0.2, 0.2, 0.2]   # 从考试（重新开始）
    ])
    
    # 定义奖励函数
    rewards = np.array([15.0, -8.0, -2.0, 25.0])  # 专心学习+15, 分心-8, 睡觉-2, 考试+25
    
    # 折扣因子
    gamma = 0.9
    
    print("状态定义:")
    for state, desc in state_descriptions.items():
        print(f"  • {state}: {desc}")
    
    print(f"\n奖励设计理念:")
    print(f"  • 专心学习 (+15): 获得知识，长期收益高")
    print(f"  • 分心 (-8): 浪费时间，机会成本高")
    print(f"  • 睡觉 (-2): 虽然必要，但学习时间减少")
    print(f"  • 考试 (+25): 检验学习成果，获得成就感")
    
    # 创建MRP
    mrp = MarkovRewardProcessDetailed(states, transition_matrix, rewards, gamma, state_descriptions)
    
    return mrp


def demonstrate_mrp_concepts():
    """演示马尔可夫奖励过程的核心概念"""
    print("\n" + "🎓" * 40)
    print("马尔可夫奖励过程 (MRP) 核心概念演示")
    print("🎓" * 40)
    
    # 创建示例MRP
    mrp = create_student_learning_mrp()
    
    # 打印详细信息
    mrp.print_detailed_summary()
    
    # 计算价值函数 - 解析解
    print("\n" + "="*60)
    print("🧮 第一步：计算价值函数 - 解析解方法")
    print("="*60)
    V_analytical = mrp.compute_value_function_analytical()
    
    # 计算价值函数 - 迭代解
    print("\n" + "="*60)
    print("🔄 第二步：计算价值函数 - 价值迭代方法")
    print("="*60)
    V_iterative = mrp.compute_value_function_iterative(max_iterations=100, tolerance=1e-10)
    
    # 分析平稳分布
    print("\n" + "="*60)
    print("📊 第三步：分析马尔可夫链的平稳分布")
    print("="*60)
    stationary_dist = mrp.analyze_stationary_distribution()
    
    # 创建可视化
    print("\n" + "="*60)
    print("📈 第四步：生成可视化分析")
    print("="*60)
    visualizer = MRPVisualizer(mrp)
    
    print("正在生成可视化图表...")
    
    # 1. 状态转移图
    print("1. 状态转移图")
    visualizer.plot_state_transition_graph()
    
    # 2. 转移矩阵热力图
    print("2. 转移概率矩阵热力图")
    visualizer.plot_transition_matrix_heatmap()
    
    # 3. 价值函数比较
    print("3. 价值函数比较图")
    visualizer.plot_value_function_comparison()
    
    # 4. 奖励函数
    print("4. 奖励函数图")
    visualizer.plot_reward_function()
    
    # 5. 回合模拟
    print("5. 回合模拟分析")
    returns, lengths = visualizer.plot_episode_simulation('专心学习', n_episodes=30, max_steps=20)
    
    # 最终总结
    print("\n" + "="*80)
    print("📋 马尔可夫奖励过程分析总结")
    print("="*80)
    
    print(f"\n🎯 关键发现:")
    if V_analytical is not None:
        best_state = mrp.states[np.argmax(V_analytical)]
        worst_state = mrp.states[np.argmin(V_analytical)]
        print(f"  • 最优状态: {best_state} (V = {np.max(V_analytical):.3f})")
        print(f"  • 最差状态: {worst_state} (V = {np.min(V_analytical):.3f})")
    
    if stationary_dist is not None:
        most_likely_state = mrp.states[np.argmax(stationary_dist)]
        print(f"  • 长期最可能状态: {most_likely_state} (π = {np.max(stationary_dist):.3f})")
    
    print(f"\n📊 模拟验证:")
    print(f"  • 平均折扣回报: {np.mean(returns):.3f}")
    print(f"  • 回报标准差: {np.std(returns):.3f}")
    print(f"  • 平均回合长度: {np.mean(lengths):.1f} 步")
    
    print(f"\n💡 教学要点:")
    print(f"  1. 马尔可夫性质：未来只依赖当前状态")
    print(f"  2. 价值函数：衡量状态的长期价值")
    print(f"  3. 贝尔曼方程：价值函数的递归关系")
    print(f"  4. 折扣因子：平衡即时奖励和未来奖励")
    print(f"  5. 收敛性：迭代算法的数学保证")


if __name__ == "__main__":
    # 运行完整的MRP概念演示
    demonstrate_mrp_concepts()