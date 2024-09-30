import random
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.llms import Ollama

# Uncomment the following line to use an example of a custom tool
from src.doc_gen.tools.custom_tool import MedicalDocumentTemplateTool, MedicalDialogueSampleTool
# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool

# local_llm = "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
local_llm1 = "ollama/qwen2.5:latest"
local_llm2 = "ollama/Gemma-Ko-Merge:latest"

#TODO 1: 치의학 상황에 대한 대화 발생시키는 task, 이를 위한 tools
#TODO 2: 대화 요약 및 인텐트 분석하는 task
#TODO 3: 의학 보고서 작성하는 task


@CrewBase
class DocGenCrew():
	"""DocGen crew"""

	@agent
	def conversation_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['Conversation_Generator'],
			# tools=[MedicalDialogueSampleTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=local_llm2
		)

	@agent
	def medical_intent_extractor(self) -> Agent:
		return Agent(
            config=self.agents_config['Medical_Intent_Extractor'],
            verbose=True,
            llm=local_llm2
        )

	@agent
	def medical_interpreter(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Interpreter'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=local_llm2
		)

	@agent
	def medical_report_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Report_Writer'],
			# tools=[MedicalDocumentTemplateTool(),],
			verbose=True,
			llm=local_llm2
		)

	@task
	def conversation_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config['conversation_generation_task'],
			output_file='outputs/dialouge.txt'
		)
	
	@task
	def intent_extraction_task(self) -> Task:
		return Task(
	        config=self.tasks_config['intent_extraction_task'],
	        output_file='outputs/intents.txt'
		)
	
	@task
	def consultation_analysis_task(self) -> Task:
		# self.tasks_config['consultation_analysis_task'].update({
		# 	"context": ["conversation_generation_task",]
		# })
		return Task(
			config=self.tasks_config['consultation_analysis_task'],
			output_file='outputs/summary.txt'
		)

	@task
	def report_generation_task(self) -> Task:
		# self.tasks_config['report_generation_task'].update({
		# 	"context": ["conversation_generation_task", "consultation_analysis_task"]
		# })
		return Task(
			config=self.tasks_config['report_generation_task'],
			output_file='outputs/report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the DocGen crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			verbose=True,
			planning=True,
			share_crew=True,
			# memory=True,
			planning_llm=local_llm1,
			manager_agent=Agent(
				role='작업 매니저',
				goal='전체 작엄 매니지먼트 전문가',
				backstory="""
				크루들 간의 원활한 업무가 진행되도록 관리, 감독합니다.
				""",
				verbose=True,
				allow_delegation=True,
				llm=local_llm1,
			)
		)
