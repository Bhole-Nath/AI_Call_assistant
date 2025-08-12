from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from typing import List

from tools.custom_tools import STTTool, DataExtractorTool, TTSTool

@CrewBase
class AiCallbotCrew():
    """AiCallbotCrew crew"""

    agents: List[Agent]
    tasks: List[Task]

    # point to YAML
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    _local_llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

    @agent
    def STT_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['STT_Agent'], # type: ignore[index]
            verbose=True,
            tools=[STTTool()],
            llm=self._local_llm)

    @agent
    def LLM_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['LLM_Agent'], # type: ignore[index]
            verbose=True,
            llm=self._local_llm)
    
    @agent
    def Data_Extractor_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Data_Extractor_Agent'],
            verbose=True,
            tools=[DataExtractorTool()],
            llm=self._local_llm)
              
    @agent
    def TTS_Agent(self) -> Agent:
        return Agent(
            config=self.agents_config['TTS_Agent'],
            verbose=True,
            tools=[TTSTool()],
            llm=self._local_llm)

    @task
    def STT_Task(self) -> Task:
        return Task(
            config=self.tasks_config['STT_Task'],
        )

    @task
    def LLM_task(self) -> Task:
        return Task(
            config=self.tasks_config['LLM_task'],
        )

    @task
    def TTS_task(self) -> Task:
        return Task(
            config=self.tasks_config['TTS_task'],
        )

    @task
    def Data_Extractor_Task(self) -> Task:
        return Task(
            config=self.tasks_config['Data_Extractor_Task'],
            output_file='conversation.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiCallbotCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
