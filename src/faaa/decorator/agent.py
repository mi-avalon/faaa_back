# Copyright 2024 TsumiNa.
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Coroutine

from pydantic import BaseModel

from faaa.core.llm import LLMClient
from faaa.core.tool_schema import Tool
from faaa.util import generate_id


class AgentError(Exception):
    def __init__(self, message: str | BaseException | None = None):
        self.message = f"Agent error: {message}"
        super().__init__(self.message)


class _AgentToolSchema(BaseModel):
    """
    A schema class for representing an agent with various attributes.

    Attributes:
        func (Callable): The function associated with the agent.
        entry_points (str): The entry points for the agent.
        code_id (str): The unique identifier for the code.
        is_async (bool): Indicates whether the agent operates asynchronously.
        tool (Tool): The tool associated with the agent.
    """

    func: Callable
    entry_points: str
    code_id: str
    tool: Tool


class Agent:
    def __init__(self, *, prefix_path: str = "/agent/v1", **kwargs):
        self._llm_client = None
        self._tools: dict[str, _AgentToolSchema] = {}
        self._prefix_path = prefix_path
        self._thread_pool_executor: ThreadPoolExecutor | None = None
        self._process_pool_executor: ProcessPoolExecutor | None = None
        self._registration_tasks: Coroutine[Any, Any, _AgentToolSchema | None] = []

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    async def register_tools(self):
        if not self._registration_tasks:
            return
        self._tools = {t.code_id: t for t in await asyncio.gather(*self._registration_tasks) if t is not None}

        self._registration_tasks.clear()  # Clear tasks after execution
        return self._tools

    def __repr__(self):
        return f"<Agent instance with schemas: {self._tools}>"

    @property
    def tools(self):
        return self._tools

    def update_config(self, **kwargs):
        """Update agent configuration with provided kwargs."""
        pass

    @staticmethod
    def _get_source_code(func: Callable) -> str:
        """Get source code of a function. This is a separate function to be picklable."""
        return inspect.getsource(func).strip()

    @classmethod
    def get_function_file_name(cls, func: Callable) -> str:
        try:
            # Get the file path where the function is located
            file_path = inspect.getfile(func)
            # Check if the file path actually exists
            if os.path.exists(file_path):
                return os.path.splitext(os.path.basename(file_path))[0]
            else:
                # Return "/" if the path does not exist
                return "/"

        except TypeError:  # Built-in functions may trigger TypeError
            # Return "/" for built-in or unknown functions
            return "/"

    async def _register_tool(
        self, original_func: Callable, wrapped_func: Callable
    ) -> _AgentToolSchema | None:
        """
        Register a function as a tool.

        Args:
            original_func: The original function (used for metadata)
            wrapped_func: The wrapped function (used for execution)
        """
        if not callable(original_func) or not callable(wrapped_func):
            raise ValueError("Both original_func and wrapped_func must be callable")

        _ = inspect.getsource(original_func).strip()
        code_id = generate_id(_)

        # Skip if already registered
        if code_id in self._tools:
            return

        # Generate tool schema in thread pool (I/O-bound)
        if self._thread_pool_executor is None:
            raise ValueError("ThreadPoolExecutor not initialized.")

        tool_schema = await self.llm_client.generate_tool_description(original_func)
        file_name = self.get_function_file_name(original_func)
        prefix_path = f"{self._prefix_path}/{file_name}"
        entry_points = f"{prefix_path}/{tool_schema.name}"

        return _AgentToolSchema(
            func=wrapped_func,  # Store wrapped function for execution
            entry_points=entry_points,
            code_id=code_id,
            tool=tool_schema,
        )

    def register(self, *, use_process=False):
        def decorator(func):
            if not callable(func):
                raise ValueError("The provided func must be a callable")

            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                wrapped = async_wrapper
            else:

                @wraps(func)
                async def sync_wrapper(*args, **kwargs):
                    loop = asyncio.get_running_loop()
                    executor = self._process_pool_executor if use_process else self._thread_pool_executor

                    if executor is None:
                        raise ValueError(
                            f"{(ProcessPoolExecutor if use_process else ThreadPoolExecutor).__name__} not initialized."
                        )
                    return await loop.run_in_executor(executor, func, *args, **kwargs)

                wrapped = sync_wrapper

            # Add registration task with both original and wrapped functions
            self._registration_tasks.append(self._register_tool(func, wrapped))
            return wrapped

        return decorator
