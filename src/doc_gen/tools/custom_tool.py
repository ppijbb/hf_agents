import random
from typing import Any, Callable, Optional, Type
from crewai_tools import BaseTool as CrewBaseTool
from crewai_tools import (SerperDevTool, ScrapeWebsiteTool, DallETool,
                          WebsiteSearchTool, SeleniumScrapingTool, tool)
from crewai.tasks.task_output import TaskOutput
from crewai.tasks.conditional_task import ConditionalTask

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


class CustomTool(CrewBaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )

    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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

class PubmedTool(PubmedQueryRun):
    def run(
        self,
        *args:Any,
		**kwargs: Any) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args,)

class ArxivTool(ArxivQueryRun):
    def run(
        self,
        *args:Any,
		**kwargs: Any) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args,)

class WebSearchTool(DuckDuckGoSearchRun):
    def run(
        self,
        *args:Any,
		**kwargs: Any) -> Any:
        print(f"Using Tool: {self.name}")
        return self._run(*args,)
