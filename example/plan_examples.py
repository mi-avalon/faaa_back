"""使用FaaA的generate_plan功能的示例。

运行此示例前需要：
1. 在项目根目录创建.env文件，包含：
   OPENAI_API_KEY="your-api-key"
   OPENAI_BASE_URL="https://openrouter.ai/api/v1"

2. 将"your-api-key"替换为你的实际OpenAI API密钥
"""

import asyncio

from agent_functions import calculate_fibonacci, fetch_delayed_greeting, prime_factors

from faaa.app import FaaA
from faaa.decorator.agent import Agent


async def main():
    # 创建并配置agents
    agent = Agent()

    agent.register(use_process=False)(calculate_fibonacci)
    agent.register(use_process=True)(prime_factors)
    agent.register()(fetch_delayed_greeting)

    # 初始化FaaA
    async with FaaA() as fa:
        # 包含我们的agent
        fa.include_agents(agent)
        # 注册所有工具
        await fa.register_agents()

        print("简单示例：计算斐波那契数列")
        # 使用单个工具的简单查询
        simple_plans = await fa.generate_plan("我需要计算斐波那契数列中的第10个数字，请告诉我具体步骤。")
        for simple_plan in simple_plans:
            print(f"\n计划ID: {simple_plan.id}")
            print(f"描述: {simple_plan.description}")
            print(f"推荐分数: {simple_plan.recommendation_score}")
            print("\n步骤:")
            for i, step in enumerate(simple_plan.steps or [], 1):
                print(f"\n步骤 {i}:")
                print(f"描述: {step.description}")
                print(f"工具: {step.suggested_tool}")
                print(f"查询: {step.sub_query}")
                print(f"解释: {step.explanation}")

            print("\n" + "=" * 50 + "\n")

        print("带重试的示例：延迟问候")
        # 展示可能失败操作的重试示例
        retry_plans = await fa.generate_plan("""
        我需要向小明发送一个延迟2秒的问候。
        由于网络操作可能会失败，如果失败的话应该重试。
        """)

        for retry_plan in retry_plans:
            print(f"\n计划ID: {retry_plan.id}")
            print(f"描述: {retry_plan.description}")
            print(f"推荐分数: {retry_plan.recommendation_score}")
            print("\n步骤:")
            for i, step in enumerate(retry_plan.steps or [], 1):
                print(f"\n步骤 {i}:")
                print(f"描述: {step.description}")
                print(f"工具: {step.suggested_tool}")
                print(f"查询: {step.sub_query}")
                print(f"解释: {step.explanation}")
                print(f"重试次数: {step.retry}")

            print("\n" + "=" * 50 + "\n")

        print("工具不足的示例")
        # 展示当工具不足时的工具推荐示例
        insufficient_plan = (
            await fa.generate_plan("""
        我需要完成以下计算：
        1. 计算25的平方根
        2. 将结果四舍五入到小数点后3位
        """)
        )[0]

        print(f"\n计划ID: {insufficient_plan.id}")
        print(f"描述: {insufficient_plan.description}")
        print(f"推荐分数: {insufficient_plan.recommendation_score}")

        if insufficient_plan.recommendation_tools:
            print("\n需要的推荐工具:")
            for tool in insufficient_plan.recommendation_tools:
                print(f"\n工具: {tool.name}")
                print(f"描述: {tool.description}")
                print(f"原因: {tool.reason}")
                print("参数:")
                for param in tool.parameters:
                    print(f"- {param.name}: {param.type} ({'必需' if param.required else '可选'})")
                    print(f"  描述: {param.description}")
        else:
            print("应该没有可用步骤，但是:")
            print("\n步骤:")
            for i, step in enumerate(insufficient_plan.steps or [], 1):
                print(f"\n步骤 {i}:")
                print(f"描述: {step.description}")
                print(f"工具: {step.suggested_tool}")
                print(f"查询: {step.sub_query}")
                print(f"解释: {step.explanation}")
                print(f"重试次数: {step.retry}")

        print("\n" + "=" * 50 + "\n")

        print("复杂示例：数学分析")
        # 组合多个工具的复杂查询
        complex_plans = await fa.generate_plan("""
        我需要进行以下数学分析：
        1. 计算斐波那契数列的第15个数字
        2. 找出这个数字的所有质因数
        3. 对每个质因数，发送一个以该数字作为延迟秒数的问候消息
        """)

        for complex_plan in complex_plans:
            print(f"\n计划ID: {complex_plan.id}")
            print(f"描述: {complex_plan.description}")
            print(f"推荐分数: {complex_plan.recommendation_score}")
            print("\n步骤:")
            for i, step in enumerate(complex_plan.steps or [], 1):
                print(f"\n步骤 {i}:")
                print(f"描述: {step.description}")
                print(f"工具: {step.suggested_tool}")
                print(f"查询: {step.sub_query}")
                print(f"解释: {step.explanation}")

            if complex_plan.recommendation_tools:
                print("\n推荐工具:")
                for tool in complex_plan.recommendation_tools:
                    print(f"- {tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
