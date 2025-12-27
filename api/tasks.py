"""
Asynchronous Celery tasks for AgentForge.
"""
import asyncio
from core.celery_app import celery_app
from core.agents import create_swarm_graph
from utils.logger import get_logger

logger = get_logger(__name__)

@celery_app.task(name="run_swarm_task")
def run_swarm_task(task_topic: str, provider: str = "gemini"):
    """
    Celery task to run a swarm workflow.
    Since LangGraph ainvoke is async, we run it in an event loop.
    """
    logger.info(f"STARTING ASYNC TASK | Swarm: {task_topic}")
    
    async def _run():
        graph = create_swarm_graph(provider=provider)
        result = await graph.ainvoke({"topic": task_topic})
        return result

    # Run the async code in a synchronous Celery worker
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_run())
    
    logger.info(f"COMPLETED ASYNC TASK | Swarm: {task_topic}")
    return result
