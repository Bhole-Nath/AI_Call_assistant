from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class AiCallbotCrew():
    """AiCallbotCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def STT_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['STT_Agent'], # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def LLM_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['LLM_Agent'], # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()]
        )
    
    @agent
    def Data_Extractor_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Data_Extractor_Agent'], # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()]
        )
    
    @agent
    def TTS_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['TTS_Agent'], # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()]
        )

    @task
    def STT_Task(self) -> Task:
        return Task(
            config=self.tasks_config['STT_Task'], # type: ignore[index]
        )

    @task
    def LLM_task(self) -> Task:
        return Task(
            config=self.tasks_config['LLM_task'], # type: ignore[index]
        )

    @task
    def TTS_task(self) -> Task:
        return Task(
            config=self.tasks_config['TTS_task'], # type: ignore[index]
        )

    @task
    def Data_Extractor_Task(self) -> Task:
        return Task(
            config=self.tasks_config['Data_Extractor_Task'], # type: ignore[index]
            output_file='conversation.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiCallbotCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical
        )
