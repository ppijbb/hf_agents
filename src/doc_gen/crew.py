from datetime import datetime
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DOCXSearchTool
from langchain_community.tools import PubmedQueryRun
from langchain_community.llms import Ollama
from langchain.agents import Tool

# Uncomment the following line to use an example of a custom tool
from src.doc_gen.tools.custom_tool import (MedicalDocumentTemplateTool, MedicalDialogueSampleTool, 
                                           final_parser,
										   WebSearchTool, PubmedTool, ArxivTool, CrewQuery)
# Check our tools documentations for more information on how to use them
from crewai_tools import SerperDevTool

# local_llm = "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
local_llm1 = "ollama/qwen2.5:latest"
# local_llm2 = "ollama/Gemma-Ko-Merge:latest"

# ollama engine
# local_llm2 = LLM(
# 	model="ollama/Gemma-Ko-Merge:latest",
# 	temperature=0.5,
# 	max_tokens=8129,
# 	base_url="http://localhost:11434",
# )

# vLLM engine
local_llm2 = LLM(
	model=f"ollama/{os.getenv('VLLM_MODEL')}",
	temperature=0.1,
	max_tokens=8192,
	# base_url="http://localhost:8000/v1",
 	# api_key="NOT A REAL KEY",
)
pub = PubmedQueryRun()
PubTool = Tool(
	name=pub.name,
	description=pub.description,
    func=pub.run
)


@CrewBase
class DocGenCrew():
	"""DocGen crew"""

	def _get_time_now(self):
		return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	
	@agent
	def domain_searcher(self) -> Agent:
		return Agent(
			config=self.agents_config['Domain_Searcher'],
			tools=[
				PubTool,
       			# PubmedTool(),
          		WebSearchTool()
            ], # Example of custom tool, loaded on the beginning of file
   			step_callback=final_parser,
			verbose=True,
			max_iter=1,
			llm=local_llm2
		)

	@agent
	def conversation_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['Conversation_Generator'],
			# tools=[MedicalDialogueSampleTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			max_iter=5,
     		step_callback=final_parser,
			llm=local_llm2
		)

	@agent
	def medical_intent_extractor(self) -> Agent:
		return Agent(
            config=self.agents_config['Medical_Intent_Extractor'],
            verbose=True,
			max_iter=5,
            step_callback=final_parser,
            llm=local_llm2
        )

	@agent
	def medical_interpreter(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Interpreter'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			max_iter=5,
   			step_callback=final_parser,
			llm=local_llm2
		)

	@agent
	def medical_report_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Report_Writer'],
			# tools=[MedicalDocumentTemplateTool(),],
			verbose=True,
			max_iter=5,
   			step_callback=final_parser,
			llm=local_llm2
		)

	@task
	def domain_searching_task(self) -> Task:
		return Task(
			config=self.tasks_config['domain_searching_task'],
			output_file=f"outputs/domain_report.md",
			# output_file=f'outputs/domain_report_{self._get_time_now()}.txt'
		)	

	@task
	def conversation_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config['conversation_generation_task'],
   			context=[
    	      self.domain_searching_task()
	         ],
			output_file=f"outputs/dialouge.txt",
			# output_file=f'outputs/dialouge_{self._get_time_now()}.txt'
		)

	@task
	def intent_extraction_task(self) -> Task:
		return Task(
	        config=self.tasks_config['intent_extraction_task'],
         	context=[
              self.conversation_generation_task()
              ],
			output_file=f"outputs/intents.md",
	        # output_file=f'outputs/intents_{self._get_time_now()}.txt'
		)

	@task
	def consultation_analysis_task(self) -> Task:
		# self.tasks_config['consultation_analysis_task'].update({
		# 	"context": ["conversation_generation_task",]
		# })
		return Task(
			config=self.tasks_config['consultation_analysis_task'],
   			context=[
          		self.conversation_generation_task(),
				self.intent_extraction_task()
    		],
			output_file=f"outputs/summary.md",
			# output_file=f'outputs/summary_{self._get_time_now()}.txt'
		)

	@task
	def report_generation_task(self) -> Task:
		# self.tasks_config['report_generation_task'].update({
		# 	"context": ["conversation_generation_task", "consultation_analysis_task"]
		# })
		return Task(
			config=self.tasks_config['report_generation_task'],
   			context=[
          		self.conversation_generation_task(),
				self.intent_extraction_task(),
				self.consultation_analysis_task()
    		],
			output_file=f"outputs/report.md",
			# output_file=f'outputs/report_{self._get_time_now()}.md'
		)
  
	@task
	def treatment_searching_task(self) -> Task:
		# self.tasks_config['report_generation_task'].update({
		# 	"context": ["conversation_generation_task", "consultation_analysis_task"]
		# })
		return Task(
			config=self.tasks_config['treatment_searching_task'],
   			context=[
          		self.conversation_generation_task(),
				self.intent_extraction_task(),
				self.consultation_analysis_task(),
				self.report_generation_task()
    		],
			output_file=f"outputs/treatment.md",
			# output_file=f'outputs/report_{self._get_time_now()}.md'
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
			# planning=True,
			share_crew=True,
   
			# memory=True,
			# planning_llm=local_llm1,
			# manager_agent=Agent(
			# 	role='작업 매니저',
			# 	goal='전체 작엄 매니지먼트 전문가',
			# 	backstory="""
			# 	크루들 간의 원활한 업무가 진행되도록 관리, 감독합니다.
			# 	""",
			# 	verbose=True,
			# 	allow_delegation=True,
			# 	llm=local_llm1,
			# )
		)
