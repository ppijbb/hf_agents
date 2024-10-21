import os
import datetime
from typing import Union
from tqdm.auto import tqdm
from crewai import Agent, Task, Crew, Process, Pipeline, LLM
from crewai.agents.parser import CrewAgentParser
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai_tools import (SerperDevTool, ScrapeWebsiteTool, DallETool,
                          WebsiteSearchTool, SeleniumScrapingTool, tool)
from crewai_tools.tools.base_tool import BaseTool as CrewBaseTool
from crewai.tasks.task_output import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask

from langchain_community.tools import (WikipediaQueryRun, PubmedQueryRun,
                                       YouTubeSearchTool, OpenWeatherMapQueryRun)
from langchain_community.utilities import SearxSearchWrapper

from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, SearxSearchRun
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.google_trends.tool import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.llms import Ollama
from langchain.tools import BaseTool
from langchain.tools import BaseTool as LangBaseTool
from pytrends.request import TrendReq

from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options

import pandas as pd
from pydantic import BaseModel, Field

from typing import Any, Optional, Type
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from pydantic import BaseModel, field_validator

os.environ['VLLM_MODEL'] = "Gemma-Ko-Merge"
# vLLM engine
en_llm = LLM(
	model=f"ollama/{os.getenv('VLLM_MODEL')}",
	temperature=0.1,
	max_tokens=2048,
	# base_url="http://localhost:8000/v1",
 	# api_key="NOT A REAL KEY",
)

ch_llm = en_llm

class FinallParser(CrewAgentExecutor):
    def _format_answer(self, answer: str) -> Union[AgentAction, AgentFinish]:
        return CrewAgentParser(agent=self.agent).parse(answer)

class CrewQuery(BaseModel):
    query: str = Field(...)

class CrewQuery(BaseModel):
    query: str = Field(...)

    @field_validator("query", mode="before")
    @classmethod
    def dict_to_str(cls, v) -> str:
        print(v)
        if isinstance(v, dict):
            if "query" in v:
                v = v["query"]
        return str(v["title"]) if isinstance(v, dict) else str(v)

class QueryProcessor(CrewBaseTool):
    runnable_tool: LangBaseTool | CrewBaseTool
    
    def _check_query(self, query: str) -> bool:
        if isinstance(query, dict):
            if "query" in query:
                if isinstance(query["query"], dict):
                    if "title" in query["query"]:
                        query["query"] = query["query"]["title"]
    
    def __call__(self, *args, **kwargs) -> Any:
        print(100)
        return self.run(*args, **kwargs)
    
    def _run(self, query:str, callbacks=None, *args, **kwargs) -> Any:
        print(f"Using Tool: {self.name}")
        print(type(query), query)
        data = self.runnable_tool.run(query)
        print(data)
        return data

class PubmedTool(QueryProcessor):
    runnable_tool: LangBaseTool | CrewBaseTool = PubmedQueryRun()
    name: str = runnable_tool.name
    description: str = runnable_tool.description
    args_schema: Type[BaseModel] = CrewQuery

class ArxivTool(QueryProcessor):
    runnable_tool: LangBaseTool | CrewBaseTool = ArxivQueryRun()
    name: str = runnable_tool.name
    description: str = runnable_tool.description
    args_schema: Type[BaseModel] = CrewQuery

class WebSearchTool(QueryProcessor):
    runnable_tool: LangBaseTool | CrewBaseTool = DuckDuckGoSearchRun()
    name: str = runnable_tool.name
    description: str= runnable_tool.description
    args_schema: Type[BaseModel] = CrewQuery

search = DuckDuckGoSearchRun()
webtool = Tool(
  name=search.name,
  description=search.description,
  func=search.run,
)
pubmed = PubmedQueryRun()
pubtool = Tool(
  name=pubmed.name,
  description=pubmed.description,
  func=pubmed.run)

agent_tools = [
      # DuckDuckGoSearchRun(),
      # WebSearchTool()
      # webtool,
      pubtool
    #   ArxivQueryRun(),
    #   PubmedTool(),
    #   PubmedQueryRun(args_schema=CrewQuery),
    #   YouTubeSearchTool(args_schema=CrewQuery)
    ]
def final_parser(answer: AgentAction):
  fixed_prompt = "Once all necessary information is gathered:"
  print(answer.thought)
  if "Final Answer:" in answer.result and "Action:" in answer.result:
    answer.result = fixed_prompt + "\n\n" + answer.result.split(fixed_prompt)[1]
    print(answer.result)

search_agent = Agent(
  role='Search Engine',
  goal='Search for specific treatment effect reports on dental treatment using the agent prompt.',
  backstory="""Find the medical infomations with tools.""",
  verbose=True,
  step_callback=final_parser,
  llm=en_llm, # ollama_openhermes,
  allow_delegation=False,
  tools=agent_tools,
#   tools=agent_tools,
)

summ_agent = Agent(
  role='Summarizer',
  goal='Summarize found document',
  backstory="""Summarize as abstraction, report, short blog etc...""",
  verbose=True,
  llm=ch_llm, # ollama_solar,
  allow_delegation=True,
  # tools=agent_tools,
)

team_manager = Agent(
  role='작업 매니저',
  goal='전체 작엄 매니지먼트 전문가',
  backstory="""크루들 간의 원활한 업무가 진행되도록 관리, 감독합니다.""",
  verbose=True,
  llm=en_llm, # ollama_solar,
  allow_delegation=True,
)

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
      Task completed!
      Task: {output.description}
      Output: {output.raw_output}
    """)

# Create tasks for your agents

def get_current_time_text():
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 함수를 사용하여 현재 시간을 텍스트로 가져옵니다.
current_time_text = get_current_time_text()

search_task = Task(
  description=f"""
  today is {get_current_time_text()}.
  I need to find medical infomations with detailed surgical procedure
  Topic will updated every step i generate, modified by me.
  Step by step, will update more detail query topic.
  Let's Start Task~
  - start topic : Treatment of an ambush permanent tooth surgical procedure
  (If Final Answer, don't generate Action in sentence.)
""",
  expected_output='infomation',
  allow_delegation=False,
#   output_pydantic=AgentSharedForm,
  agent=search_agent,
  max_iter=1,
)

summ_task = Task(
  description="""
  make summarization of paper
""",
  expected_output='abstract of paper',
  agent=summ_agent,
  allow_delegation=False,
#   output_pydantic=AgentRerunForm,
  max_iter=1,
  context=[search_task]
)


# Instantiate your crew with a sequential process
transcript_crew = Crew(
  step_callback="",
  agents=[
      search_agent,
      summ_agent],
  tasks=[
      search_task,
      summ_task],
#   full_output=False,
#   planning=True,
  function_calling_llm=en_llm,
  verbose=True, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
  process=Process.sequential, # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
  manager_agent=team_manager,
  planning_llm=en_llm,
  language="ko"
)

# output_crew = Crew(
#   agents=[ labeler,],
#   tasks=[making_data],
#   verbose=False, # Crew verbose more will let you know what tasks are being worked on, you can set it to 1 or 2 to different logging levels
#   process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
# )

# pipeline = Pipeline(
#     stages=[label_crew, output_crew]
# )

# # Get your crew to work!
# result = await pipeline.process_single_kickoff(
#     dict(
#         topic="서울의 봄"
#     )
# )

sentence_list = [
    # "음, 그건 꽤 흥미로운 주제네요. 제 견해로는 맥스웰 방정식이 양자전기역학의 고전적 극한으로 해석될 수 있다고 봅니다. 특히 광자의 개념을 도입하면 두 이론 간의 연결고리가 더 명확해지죠.",
    "대칭적인 형태로 이렇게 제작이 될 거고, 위 턱의 어금니쪽 같은 경우는 광대뼈 안쪽에 상악동이라고 하는 공기주머니가 있어요. "
    # "아 진짜 아무것도 아니야. 그냥 넘어가.",
    # "주말에는 좀 그렇고... 다음 주 수요일은 어때?",
]

filename = "your_model.pkl"
# train model in process
# try:
#     transcript_crew.train(
#         n_iterations=3,
#         inputs={"sentence": random.choice(sentence_list)},
#         filename=filename)
# except Exception as e:
#     raise Exception(f"An error occurred while training the crew: {e}")


result = transcript_crew.kickoff()

print("######################")
print(result)
