# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT


import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel

from faaa.core.llm import LLMClient
from faaa.core.tool_schema import ToolParameter
from faaa.decorator.agent import Agent, _AgentToolSchema
from faaa.prompt import DYNAMIC_PLAN_INSTRUCTION
from faaa.util import generate_id, pydantic_to_yaml


class FAError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"FA error: {message}"
        super().__init__(self.message)


class PlanStep(BaseModel):
    """
    PlanStep represents a step in a plan with detailed information.

    Attributes:
        description (str): A short description of the step.
        suggested_tool (str): The tool suggested for this step.
        sub_query (str): The sub query for this step.
        explanation (str): Explanation of why this step is needed.
        retry (int): The number of retries for this step.
    """

    description: str  # A short description of the step
    suggested_tool: str  # The tool suggested for this step
    sub_query: str  # The sub query for this step
    explanation: str  # Explanation of why this step is needed
    retry: int  # The number of retries for this step


class RecommendationTool(BaseModel):
    """
    RecommendationTool is a model representing a recommendation tool with its associated attributes.

    Attributes:
        name (str): The name of the recommendation tool.
        description (str): A brief description of the recommendation tool.
        reason (str): The reason or purpose for the recommendation tool.
        parameters (list[ToolParameter]): A list of parameters associated with the recommendation tool.
    """

    name: str
    description: str
    reason: str
    parameters: list[ToolParameter]


class DynamicPlan(BaseModel):
    """
    DynamicPlan represents a dynamic plan with a description, steps, recommendation tools, and a recommendation score.

    Attributes:
        description (str): A description of the overall plan.
        steps (list[PlanStep]): A list of steps in the plan.
        recommendation_tools (list[RecommendationTool]): The recommended tools for the plan.
        recommendation_score (float): The recommendation score for the plan.
    """

    description: str  # A description of the overall plan
    steps: list[PlanStep]  # A list of steps in the plan
    recommendation_tools: list[RecommendationTool]  # The recommended tool for the plan
    recommendation_score: float  # The recommendation score for the plan


class DynamicPlanContainer(BaseModel):
    """
    A container for dynamic plans.

    Attributes:
        plans (list[DynamicPlan]): A list of dynamic plans.
    """

    plans: list[DynamicPlan]


class DynamicPlanTracer(DynamicPlan):
    """
    DynamicPlanTracer represents a dynamic plan with a unique identifier, description, steps,
    recommended tools, and a recommendation score. It also includes a trace of the plan generation process.

    Attributes:
        id (str): A unique identifier for the plan.
        n_execution (int): The number of executions for the plan.
        parent_id (str): The parent plan's identifier.
    """

    id: str  # A unique identifier for the plan
    n_execution: int = 0  # The number of executions for the plan
    parent_id: str | None = None  # The parent plan's identifier


class FaaA:
    def __init__(self, *, max_thread_workers: int | None = None):
        self._agents: dict[str, _AgentToolSchema] = {}  # Stores registered tools
        self._agent_list: list[Agent] = []  # Stores agents pending registration
        self._fastapi_app = FastAPI()
        self._llm_client = LLMClient()

        self._thread_executor = (
            ThreadPoolExecutor(max_workers=max_thread_workers) if max_thread_workers else ThreadPoolExecutor()
        )
        self._process_executor = (
            ProcessPoolExecutor(os.cpu_count() - 1) if os.cpu_count() >= 2 else ProcessPoolExecutor()
        )

    def include_agents(self, *agent: Agent, **kwargs):
        """
        Include agents for later registration.
        Updates agent configs and stores them for processing during register_agent.

        Args:
            *agent: Agent instances to include
            **kwargs: Configuration parameters to update agents with
        """
        for a in agent:
            a.update_config(**kwargs)
            self._agent_list.append(a)

    async def register_agents(self):
        """
        Register all tools from agents in _agent_list.
        Merges registered tools into self._agents and clears _agent_list.
        """
        # Register tools for all agents
        for agent in self._agent_list:
            agent._thread_pool_executor = self._thread_executor
            agent._process_pool_executor = self._process_executor
            # Merge agent's tools into self._agents
            self._agents.update(await agent.register_tools())

        # Clear agent list after registration
        self._agent_list.clear()

    def __repr__(self):
        if not self._agents:
            return "FA(agents={})"
        agents_repr = "\n".join(f"{name}: {agent}" for name, agent in self._agents.items())
        return f"FA(agents={{\n{agents_repr}\n}})"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._thread_executor.shutdown(wait=True)
        self._process_executor.shutdown(wait=True)

    async def generate_plan(self, query: str):
        if not self._agents:
            id = generate_id("No agents available")
            return DynamicPlanTracer(
                id=id,
                description="No agents available",
                steps=None,
                recommendation_tools=None,
                recommendation_score=0.0,
            )

        query = f"""
<Query>\n{query}\n</Query>
{'\n'.join(['<Tool>\n'+pydantic_to_yaml(s.tool)+'</Tool>' for s in self._agents.values()])}
""".strip()
        query = [{"role": "system", "content": DYNAMIC_PLAN_INSTRUCTION}, {"role": "user", "content": query}]
        DPs = await self._llm_client.structured_output(
            query,
            structured_outputs=DynamicPlanContainer,
            max_try=1,
            max_tokens=10000,
            model="anthropic/claude-3.5-haiku",
        )

        return [
            DynamicPlanTracer(id=generate_id(plan.description), **plan.model_dump()) for plan in DPs.plans
        ]
