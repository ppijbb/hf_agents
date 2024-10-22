import random
from typing import Any, Callable, Optional, Type
from crewai_tools import BaseTool as CrewBaseTool
from crewai_tools import (SerperDevTool, ScrapeWebsiteTool, DallETool,
                          WebsiteSearchTool, SeleniumScrapingTool, tool)
from crewai.tasks.task_output import TaskOutput
from crewai.agents.parser import AgentAction
from crewai.tasks.conditional_task import ConditionalTask

from langchain.tools import BaseTool as LangBaseTool
from langchain_community.tools import (WikipediaQueryRun, PubmedQueryRun,
                                       YouTubeSearchTool, OpenWeatherMapQueryRun)
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchRun, SearxSearchRun
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.google_trends.tool import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from src.doc_gen.tools.tool_data import DIALOGUE_SAMPLES

from pydantic import BaseModel, Field, field_validator


def final_parser(answer: AgentAction):
    
    fixed_prompt = "Once all necessary information is gathered:"
    try:
        if "Final Answer:" in answer.result and "Action" in answer.result:
            processed = fixed_prompt + "\n\n" + answer.result.split(fixed_prompt)[1]
            answer.result = processed
            answer.text = processed
    except:
        print(answer.text)

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

class CustomTool(CrewBaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args, **kwargs)


class MedicalDocumentTemplateTool(CrewBaseTool):
	name: str = "MedicalDocumentTemplateTool"
	description: str = (
		"Give Report Format."
		)

	def _run(self, tool_input: str) -> str:
        # 다양한 보고서 유형에 따른 템플릿 반환
		return """
        # Medical Report

        ## Patient Information
        - Name: {name}
        - Age: {age}
        - Gender: {gender}

        ## Chief Complaint
        {chief_complaint}

        ## Diagnosis
        {diagnosis}

        ## Treatment Plan
        {treatment_plan}
        """

class MedicalDialogueSampleTool(CrewBaseTool):
	name: str = "MedicalDialogueSampleTool"
	description: str = (
		"Sample dialouge example for generation. Will give a random dialogue.",
		"Cannot use this tool's output as a Final Answer."
		)

	def _run(self, tool_input: str) -> str:
        # 다양한 보고서 유형에 따른 템플릿 반환
		return f"""<example>\n{random.choice(DIALOGUE_SAMPLES).strip()}\n</example>"""

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
        return self.runnable_tool.run(query)

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

