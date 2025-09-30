"""
马尔科夫奖励过程 (MRP) 课堂演示脚本

这是一个完整的课堂演示脚本，展示了MRP的所有核心概念。
适合在课堂上逐步演示，帮助学生理解MRP的数学原理和实际应用。

使用方法:
    python demo.py

作者：教学示例代码
用途：强化学习课堂教学演示
"""

import numpy as np
import matplotlib.pyplot as plt
from markov_reward_process import MarkovRewardProcess, MRPVisualizer

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')


def classroom_demo():
    """课堂演示主函数"""
    
    print("=" * 80)
    print("马尔科夫奖励过程 (Markov Reward Process) 课堂演示")
    print("=" * 80)
    print()
    
    # ========== 第一部分：理论介绍 ==========
    print("📚 第一部分：马尔科夫奖励过程理论基础")
    print("-" * 50)
    print()
    print("马尔科夫奖励过程 (MRP) 是一个四元组 (S, P, R, γ)：")
    print("• S: 状态空间 (State Space)")
    print("• P: 状态转移概率矩阵 (Transition Probability Matrix)")
    print("• R: 奖励函数 (Reward Function)")
    print("• γ: 折扣因子 (Discount Factor)")
    print()
    print("核心概念：")
    print("• 马尔科夫性质：未来只依赖于当前状态，与历史无关")
    print("• 价值函数：V(s) = E[G_t | S_t = s]，其中 G_t 是折扣回报")
    print("• 贝尔曼方程：V(s) = R(s) + γ * Σ P(s'|s) * V(s')")
    print()
    
    input("按回车键继续到实际示例...")
    
    # ========== 第二部分：简单示例 ==========
    print("\n" + "=" * 80)
    print("📊 第二部分：学生学习过程 - 经典MRP示例")
    print("=" * 80)
    
    # 创建学生MRP
    states = ['专心学习', '分心', '睡觉']
    
    print(f"\n状态空间 S = {states}")
    print("\n状态含义：")
    print("• 专心学习：学生全神贯注地学习，获得知识收益")
    print("• 分心：学生注意力不集中，学习效果差")
    print("• 睡觉：学生休息，既不学习也不娱乐")
    
    # 转移概率矩阵
    P = np.array([
        [0.7, 0.2, 0.1],  # 从专心学习
        [0.3, 0.4, 0.3],  # 从分心
        [0.2, 0.1, 0.7]   # 从睡觉
    ])
    
    print(f"\n状态转移概率矩阵 P:")
    print("     专心  分心  睡觉")
    for i, state in enumerate(states):
        print(f"{state:>4} {P[i, 0]:.1f}  {P[i, 1]:.1f}  {P[i, 2]:.1f}")
    
    # 奖励函数
    R = np.array([10.0, -5.0, 0.0])
    gamma = 0.9
    
    print(f"\n奖励函数 R = {R}")
    print("• 专心学习获得 +10 奖励（学到知识）")
    print("• 分心获得 -5 奖励（浪费时间）")
    print("• 睡觉获得 0 奖励（中性状态）")
    print(f"\n折扣因子 γ = {gamma}")
    
    input("\n按回车键创建MRP并计算价值函数...")
    
    # 创建MRP
    student_mrp = MarkovRewardProcess(states, P, R, gamma)
    
    # ========== 第三部分：价值函数计算 ==========
    print("\n" + "=" * 80)
    print("🧮 第三部分：价值函数计算")
    print("=" * 80)
    
    print("\n方法1：解析解 (Analytical Solution)")
    print("使用贝尔曼方程的矩阵形式：V = (I - γP)^(-1) * R")
    
    V_analytical = student_mrp.compute_value_function_analytical()
    
    print("\n解析解结果：")
    for i, state in enumerate(states):
        print(f"V({state}) = {V_analytical[i]:.3f}")
    
    input("\n按回车键演示迭代解...")
    
    print("\n方法2：价值迭代 (Value Iteration)")
    print("迭代公式：V_{k+1}(s) = R(s) + γ * Σ P(s'|s) * V_k(s')")
    
    V_iterative = student_mrp.compute_value_function_iterative(max_iterations=50)
    
    print(f"\n价值迭代在第 {len(student_mrp.convergence_history)} 次迭代后收敛")
    print("\n迭代解结果：")
    for i, state in enumerate(states):
        print(f"V({state}) = {V_iterative[i]:.3f}")
    
    print("\n解析解与迭代解的差异：")
    for i, state in enumerate(states):
        diff = abs(V_analytical[i] - V_iterative[i])
        print(f"{state}: {diff:.6f}")
    
    input("\n按回车键查看可视化结果...")
    
    # ========== 第四部分：可视化演示 ==========
    print("\n" + "=" * 80)
    print("📈 第四部分：可视化分析")
    print("=" * 80)
    
    visualizer = MRPVisualizer(student_mrp)
    
    print("\n正在生成可视化图表...")
    print("1. 状态转移图 - 显示状态间的转移关系")
    visualizer.plot_state_transition_graph()
    
    print("2. 转移概率矩阵热力图 - 直观显示转移概率")
    visualizer.plot_transition_matrix_heatmap()
    
    print("3. 价值函数比较 - 对比两种计算方法")
    visualizer.plot_value_function_comparison()
    
    print("4. 奖励函数可视化")
    visualizer.plot_reward_function()
    
    input("\n按回车键进行回合模拟...")
    
    # ========== 第五部分：回合模拟 ==========
    print("\n" + "=" * 80)
    print("🎮 第五部分：回合模拟与分析")
    print("=" * 80)
    
    print("\n模拟学生从'专心学习'状态开始的学习过程...")
    
    # 单次回合演示
    states_seq, rewards_seq = student_mrp.simulate_episode('专心学习', max_steps=10)
    
    print(f"\n单次回合模拟结果（10步）：")
    print("步骤  状态      奖励")
    print("-" * 25)
    for t, (state, reward) in enumerate(zip(states_seq[:-1], rewards_seq)):
        print(f"{t:>2}   {state:<8} {reward:>5.1f}")
    print(f"最终状态: {states_seq[-1]}")
    
    # 计算折扣回报
    discounted_return = sum(reward * (gamma ** t) for t, reward in enumerate(rewards_seq))
    print(f"\n折扣回报 G = {discounted_return:.3f}")
    
    # 多回合统计
    print("\n进行100次回合模拟进行统计分析...")
    returns = []
    for _ in range(100):
        _, rewards = student_mrp.simulate_episode('专心学习', max_steps=20)
        G = sum(reward * (gamma ** t) for t, reward in enumerate(rewards))
        returns.append(G)
    
    print(f"100次模拟统计结果：")
    print(f"• 平均回报: {np.mean(returns):.3f}")
    print(f"• 标准差: {np.std(returns):.3f}")
    print(f"• 理论价值 V(专心学习): {V_analytical[0]:.3f}")
    print(f"• 模拟与理论的差异: {abs(np.mean(returns) - V_analytical[0]):.3f}")
    
    # 完整的回合模拟可视化
    print("\n5. 回合模拟可视化分析")
    returns, lengths = visualizer.plot_episode_simulation('专心学习', n_episodes=20, max_steps=15)
    
    input("\n按回车键查看课堂总结...")
    
    # ========== 第六部分：课堂总结 ==========
    print("\n" + "=" * 80)
    print("📝 第六部分：课堂总结")
    print("=" * 80)
    
    print("\n今天我们学习了马尔科夫奖励过程的核心概念：")
    print("\n1. 理论基础：")
    print("   • MRP四元组：(S, P, R, γ)")
    print("   • 马尔科夫性质：无记忆性")
    print("   • 价值函数：期望折扣回报")
    
    print("\n2. 数学方法：")
    print("   • 贝尔曼方程：V(s) = R(s) + γ * Σ P(s'|s) * V(s')")
    print("   • 解析解：V = (I - γP)^(-1) * R")
    print("   • 迭代解：价值迭代算法")
    
    print("\n3. 实际应用：")
    print("   • 学生学习过程建模")
    print("   • 状态转移概率的设定")
    print("   • 奖励函数的设计")
    
    print("\n4. 计算验证：")
    print("   • 理论计算与模拟结果的一致性")
    print("   • 不同算法结果的对比")
    print("   • 收敛性分析")
    
    print("\n5. 可视化分析：")
    print("   • 状态转移图")
    print("   • 价值函数可视化")
    print("   • 回合模拟统计")
    
    print(f"\n关键数值回顾：")
    print(f"• V(专心学习) = {V_analytical[0]:.3f}")
    print(f"• V(分心) = {V_analytical[1]:.3f}")
    print(f"• V(睡觉) = {V_analytical[2]:.3f}")
    
    print("\n💡 思考题：")
    print("1. 如果提高折扣因子γ，价值函数会如何变化？")
    print("2. 如果改变转移概率，哪个状态的价值变化最大？")
    print("3. 在什么情况下解析解可能不存在？")
    
    print("\n" + "=" * 80)
    print("课堂演示结束！感谢大家的参与！")
    print("=" * 80)


def quick_demo():
    """快速演示版本，适合时间较短的课堂"""
    print("马尔科夫奖励过程 - 快速演示版")
    print("=" * 50)
    
    # 创建简单的3状态MRP
    states = ['好', '中', '差']
    P = np.array([[0.6, 0.3, 0.1],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.4, 0.5]])
    R = np.array([10, 0, -10])
    gamma = 0.9
    
    mrp = MarkovRewardProcess(states, P, R, gamma)
    mrp.print_summary()
    
    # 计算价值函数
    V = mrp.compute_value_function_analytical()
    print(f"\n价值函数：{dict(zip(states, V))}")
    
    # 简单可视化
    visualizer = MRPVisualizer(mrp)
    visualizer.plot_state_transition_graph()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        classroom_demo()