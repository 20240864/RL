"""
马尔科夫奖励过程教学示例

本文件包含多个经典的MRP教学示例，用于课堂演示和学习。

示例包括：
1. 学生马尔科夫链 - 经典的学生学习状态转移
2. 天气预测模型 - 简单的天气状态转移
3. 股票价格模型 - 金融应用示例
4. 机器维护模型 - 工程应用示例
"""

import numpy as np
import matplotlib.pyplot as plt
from markov_reward_process import MarkovRewardProcess, MRPVisualizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_student_mrp():
    """
    创建学生马尔科夫链示例
    
    这是强化学习教学中最经典的示例之一。
    学生有三种状态：
    - 专心学习 (Focus): 获得正奖励
    - 分心 (Distracted): 获得负奖励  
    - 睡觉 (Sleep): 获得零奖励
    
    返回:
        MarkovRewardProcess: 学生MRP实例
    """
    print("=" * 60)
    print("示例1: 学生马尔科夫奖励过程")
    print("=" * 60)
    print("状态说明:")
    print("- 专心学习 (Focus): 学生专注于学习，获得知识奖励")
    print("- 分心 (Distracted): 学生注意力不集中，学习效果差")
    print("- 睡觉 (Sleep): 学生休息，既不获得也不失去什么")
    print()
    
    # 定义状态
    states = ['专心学习', '分心', '睡觉']
    
    # 定义状态转移概率矩阵
    # P[i,j] = P(下一状态=j | 当前状态=i)
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # 从专心学习转移的概率
        [0.3, 0.4, 0.3],  # 从分心转移的概率
        [0.2, 0.1, 0.7]   # 从睡觉转移的概率
    ])
    
    # 定义奖励函数
    rewards = np.array([10.0, -5.0, 0.0])  # 专心学习+10，分心-5，睡觉0
    
    # 折扣因子
    gamma = 0.9
    
    # 创建MRP
    student_mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)
    
    print("转移概率解释:")
    print("- 专心学习时：70%继续专心，20%分心，10%困了去睡觉")
    print("- 分心时：30%重新专心，40%继续分心，30%累了去睡觉")
    print("- 睡觉时：20%醒来专心学习，10%醒来就分心，70%继续睡觉")
    print()
    
    return student_mrp


def create_weather_mrp():
    """
    创建天气预测MRP示例
    
    三种天气状态：
    - 晴天 (Sunny): 心情好，正奖励
    - 多云 (Cloudy): 一般，小负奖励
    - 雨天 (Rainy): 心情差，负奖励
    
    返回:
        MarkovRewardProcess: 天气MRP实例
    """
    print("=" * 60)
    print("示例2: 天气预测马尔科夫奖励过程")
    print("=" * 60)
    print("状态说明:")
    print("- 晴天 (Sunny): 天气晴朗，心情愉悦")
    print("- 多云 (Cloudy): 天气一般，心情平淡")
    print("- 雨天 (Rainy): 下雨天，心情低落")
    print()
    
    states = ['晴天', '多云', '雨天']
    
    # 天气转移概率（基于真实天气模式）
    transition_matrix = np.array([
        [0.6, 0.3, 0.1],  # 晴天后的天气概率
        [0.3, 0.4, 0.3],  # 多云后的天气概率
        [0.2, 0.5, 0.3]   # 雨天后的天气概率
    ])
    
    # 心情奖励（基于天气的心情影响）
    rewards = np.array([8.0, -1.0, -6.0])
    
    gamma = 0.85
    
    weather_mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)
    
    print("转移概率解释:")
    print("- 晴天后：60%继续晴天，30%转多云，10%转雨天")
    print("- 多云后：30%转晴天，40%继续多云，30%转雨天")
    print("- 雨天后：20%转晴天，50%转多云，30%继续下雨")
    print()
    
    return weather_mrp


def create_stock_mrp():
    """
    创建简化股票价格MRP示例
    
    三种市场状态：
    - 牛市 (Bull): 股价上涨，正收益
    - 震荡 (Sideways): 股价平稳，小收益
    - 熊市 (Bear): 股价下跌，负收益
    
    返回:
        MarkovRewardProcess: 股票MRP实例
    """
    print("=" * 60)
    print("示例3: 股票市场马尔科夫奖励过程")
    print("=" * 60)
    print("状态说明:")
    print("- 牛市 (Bull): 市场上涨趋势，投资收益为正")
    print("- 震荡 (Sideways): 市场横盘整理，收益接近零")
    print("- 熊市 (Bear): 市场下跌趋势，投资亏损")
    print()
    
    states = ['牛市', '震荡', '熊市']
    
    # 市场状态转移概率
    transition_matrix = np.array([
        [0.5, 0.3, 0.2],  # 牛市后的市场概率
        [0.25, 0.5, 0.25], # 震荡后的市场概率
        [0.2, 0.4, 0.4]   # 熊市后的市场概率
    ])
    
    # 投资收益（简化为固定值）
    rewards = np.array([15.0, 1.0, -12.0])
    
    gamma = 0.95  # 金融投资通常使用较高的折扣因子
    
    stock_mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)
    
    print("转移概率解释:")
    print("- 牛市后：50%继续牛市，30%转震荡，20%转熊市")
    print("- 震荡后：25%转牛市，50%继续震荡，25%转熊市")
    print("- 熊市后：20%转牛市，40%转震荡，40%继续熊市")
    print()
    
    return stock_mrp


def create_machine_maintenance_mrp():
    """
    创建机器维护MRP示例
    
    四种机器状态：
    - 新机器 (New): 性能最佳
    - 良好 (Good): 性能良好
    - 需维护 (Maintenance): 需要维护
    - 故障 (Broken): 机器故障
    
    返回:
        MarkovRewardProcess: 机器维护MRP实例
    """
    print("=" * 60)
    print("示例4: 机器维护马尔科夫奖励过程")
    print("=" * 60)
    print("状态说明:")
    print("- 新机器 (New): 机器状态最佳，生产效率高")
    print("- 良好 (Good): 机器状态良好，正常生产")
    print("- 需维护 (Maintenance): 机器需要维护，效率下降")
    print("- 故障 (Broken): 机器故障，无法生产")
    print()
    
    states = ['新机器', '良好', '需维护', '故障']
    
    # 机器状态转移概率
    transition_matrix = np.array([
        [0.8, 0.15, 0.05, 0.0],   # 新机器的转移概率
        [0.0, 0.7, 0.25, 0.05],   # 良好状态的转移概率
        [0.0, 0.6, 0.2, 0.2],     # 需维护状态的转移概率
        [0.0, 0.8, 0.0, 0.2]      # 故障状态的转移概率（修复后）
    ])
    
    # 生产收益（考虑维护成本）
    rewards = np.array([20.0, 15.0, -5.0, -30.0])
    
    gamma = 0.9
    
    machine_mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)
    
    print("转移概率解释:")
    print("- 新机器：80%保持新状态，15%变良好，5%需维护")
    print("- 良好状态：70%保持良好，25%需维护，5%故障")
    print("- 需维护：60%修复到良好，20%保持需维护，20%故障")
    print("- 故障状态：80%修复到良好，20%仍然故障")
    print()
    
    return machine_mrp


def demonstrate_mrp_analysis(mrp: MarkovRewardProcess, title: str):
    """
    演示MRP的完整分析过程
    
    参数:
        mrp: MRP实例
        title: 示例标题
    """
    print(f"\n{'='*20} {title} 分析 {'='*20}")
    
    # 1. 打印基本信息
    mrp.print_summary()
    
    # 2. 计算价值函数
    print("\n正在计算价值函数...")
    V_analytical = mrp.compute_value_function_analytical()
    V_iterative = mrp.compute_value_function_iterative()
    
    # 3. 比较两种方法的结果
    print(f"\n价值函数计算结果比较:")
    print(f"{'状态':<10} {'解析解':<10} {'迭代解':<10} {'差异':<10}")
    print("-" * 45)
    for i, state in enumerate(mrp.states):
        diff = abs(V_analytical[i] - V_iterative[i])
        print(f"{state:<10} {V_analytical[i]:<10.4f} {V_iterative[i]:<10.4f} {diff:<10.6f}")
    
    # 4. 创建可视化
    visualizer = MRPVisualizer(mrp)
    
    print(f"\n正在生成 {title} 的可视化图表...")
    
    # 状态转移图
    visualizer.plot_state_transition_graph()
    
    # 转移矩阵热力图
    visualizer.plot_transition_matrix_heatmap()
    
    # 价值函数比较
    visualizer.plot_value_function_comparison()
    
    # 奖励函数
    visualizer.plot_reward_function()
    
    # 回合模拟
    start_state = mrp.states[0]  # 从第一个状态开始
    returns, lengths = visualizer.plot_episode_simulation(start_state, n_episodes=10)
    
    print(f"\n回合模拟统计 (从 '{start_state}' 开始):")
    print(f"平均折扣回报: {np.mean(returns):.3f}")
    print(f"回报标准差: {np.std(returns):.3f}")
    print(f"平均回合长度: {np.mean(lengths):.1f}")


def run_all_examples():
    """运行所有教学示例"""
    print("马尔科夫奖励过程 (MRP) 教学示例集合")
    print("=" * 80)
    print("本程序将演示四个经典的MRP示例，每个示例都包含：")
    print("1. 问题描述和状态定义")
    print("2. 转移概率矩阵和奖励函数")
    print("3. 价值函数计算（解析解和迭代解）")
    print("4. 完整的可视化分析")
    print("5. 回合模拟和统计分析")
    print("=" * 80)
    
    # 示例1: 学生马尔科夫链
    student_mrp = create_student_mrp()
    demonstrate_mrp_analysis(student_mrp, "学生学习过程")
    
    # 示例2: 天气预测
    weather_mrp = create_weather_mrp()
    demonstrate_mrp_analysis(weather_mrp, "天气预测模型")
    
    # 示例3: 股票市场
    stock_mrp = create_stock_mrp()
    demonstrate_mrp_analysis(stock_mrp, "股票市场模型")
    
    # 示例4: 机器维护
    machine_mrp = create_machine_maintenance_mrp()
    demonstrate_mrp_analysis(machine_mrp, "机器维护模型")
    
    print("\n" + "=" * 80)
    print("所有示例演示完成！")
    print("=" * 80)


def interactive_mrp_explorer():
    """
    交互式MRP探索器
    允许用户选择不同的示例进行深入分析
    """
    examples = {
        '1': ('学生学习过程', create_student_mrp),
        '2': ('天气预测模型', create_weather_mrp),
        '3': ('股票市场模型', create_stock_mrp),
        '4': ('机器维护模型', create_machine_maintenance_mrp)
    }
    
    while True:
        print("\n" + "=" * 50)
        print("马尔科夫奖励过程交互式探索器")
        print("=" * 50)
        print("请选择要分析的示例:")
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        print("0. 退出")
        print("a. 运行所有示例")
        
        choice = input("\n请输入选择 (0-4, a): ").strip().lower()
        
        if choice == '0':
            print("感谢使用！再见！")
            break
        elif choice == 'a':
            run_all_examples()
        elif choice in examples:
            name, create_func = examples[choice]
            mrp = create_func()
            demonstrate_mrp_analysis(mrp, name)
        else:
            print("无效选择，请重新输入。")


if __name__ == "__main__":
    # 可以选择运行所有示例或交互式探索
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mrp_explorer()
    else:
        run_all_examples()