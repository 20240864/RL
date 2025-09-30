# RL - 马尔科夫奖励过程教学代码

这是一个详细严谨的马尔科夫奖励过程 (Markov Reward Process, MRP) 教学代码库，专为强化学习课堂教学设计。

## 📚 项目概述

本项目提供了完整的MRP实现，包括：
- 核心数学理论的严格实现
- 丰富的可视化功能
- 多个经典教学示例
- 交互式演示脚本

## 🎯 主要特性

### 核心功能
- **完整的MRP实现**：状态空间、转移概率矩阵、奖励函数、折扣因子
- **价值函数计算**：支持解析解和迭代解两种方法
- **严格的数学验证**：输入参数验证、收敛性检查
- **回合模拟**：支持单次和批量回合模拟

### 可视化功能
- **状态转移图**：直观显示状态间的转移关系
- **转移概率矩阵热力图**：清晰展示转移概率分布
- **价值函数比较图**：对比解析解和迭代解
- **收敛历史图**：展示价值迭代的收敛过程
- **回合模拟分析**：统计分析和可视化

### 教学示例
1. **学生学习过程**：经典的学生状态转移模型
2. **天气预测模型**：基于天气状态的心情奖励模型
3. **股票市场模型**：简化的金融市场状态转移
4. **机器维护模型**：工程应用中的设备状态管理

## 🚀 快速开始

### 环境要求
- Python 3.7+
- NumPy
- Matplotlib
- NetworkX
- Seaborn

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行演示
```bash
# 完整课堂演示
python demo.py

# 快速演示版本
python demo.py --quick

# 运行所有教学示例
python examples.py

# 交互式示例探索
python examples.py --interactive
```

## 📖 使用指南

### 基本用法

```python
import numpy as np
from markov_reward_process import MarkovRewardProcess, MRPVisualizer

# 定义MRP
states = ['状态1', '状态2', '状态3']
transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.1, 0.7]
])
rewards = np.array([10.0, -5.0, 0.0])
gamma = 0.9

# 创建MRP实例
mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)

# 计算价值函数
V_analytical = mrp.compute_value_function_analytical()
V_iterative = mrp.compute_value_function_iterative()

# 可视化
visualizer = MRPVisualizer(mrp)
visualizer.plot_state_transition_graph()
visualizer.plot_value_function_comparison()
```

### 高级功能

```python
# 回合模拟
states_seq, rewards_seq = mrp.simulate_episode('状态1', max_steps=20)

# 获取特定状态的价值
value = mrp.get_state_value('状态1', method='analytical')

# 打印详细信息
mrp.print_summary()
```

## 📁 项目结构

```
RL/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── markov_reward_process.py     # MRP核心实现
├── examples.py                  # 教学示例集合
└── demo.py                     # 课堂演示脚本
```

## 🎓 教学应用

### 课堂演示流程
1. **理论介绍**：MRP的数学定义和核心概念
2. **实例构建**：以学生学习过程为例构建MRP
3. **价值计算**：演示解析解和迭代解的计算过程
4. **可视化分析**：通过图表理解MRP的行为
5. **模拟验证**：通过回合模拟验证理论计算
6. **扩展讨论**：参数变化对结果的影响

### 适用场景
- 强化学习入门课程
- 概率论与随机过程课程
- 决策理论课程
- 运筹学课程

## 🔬 数学原理

### MRP定义
马尔科夫奖励过程是一个四元组 (S, P, R, γ)：
- **S**：状态空间
- **P**：状态转移概率矩阵，P[i,j] = P(S_{t+1}=j | S_t=i)
- **R**：奖励函数，R[i] = E[R_{t+1} | S_t=i]
- **γ**：折扣因子，γ ∈ [0,1]

### 价值函数
状态价值函数定义为：
```
V(s) = E[G_t | S_t = s]
```
其中 G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... 是折扣回报。

### 贝尔曼方程
```
V(s) = R(s) + γ * Σ P(s'|s) * V(s')
```

### 解析解
```
V = (I - γP)^(-1) * R
```

## 🎨 可视化示例

项目提供多种可视化功能：

1. **状态转移图**：节点表示状态，边表示转移概率
2. **热力图**：直观显示转移概率矩阵
3. **价值函数图**：比较不同计算方法的结果
4. **收敛曲线**：展示迭代算法的收敛过程
5. **模拟分析**：统计多次回合的结果分布

## 🤝 贡献指南

欢迎贡献新的教学示例、改进可视化效果或优化代码实现。请确保：
- 代码符合PEP 8规范
- 添加适当的注释和文档
- 包含必要的测试用例

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送Pull Request

---

**注意**：本代码仅用于教学目的，实际应用中可能需要根据具体需求进行调整和优化。