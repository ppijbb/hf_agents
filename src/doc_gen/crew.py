from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.llms import Ollama

# Uncomment the following line to use an example of a custom tool
# from doc_gen.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

local_llm = "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
local_llm = Ollama(model="qwen2.5:latest")

@CrewBase
class DocGenCrew():
	"""DocGen crew"""

	@agent
	def conversation_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['Conversation_Generator'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=local_llm
		)

	@agent
	def medical_interpreter(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Interpreter'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=local_llm
		)

	@agent
	def medical_report_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['Medical_Report_Writer'],
			verbose=True,
			llm=local_llm
		)

	@task
	def conversation_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config['conversation_generation_task'],
		)

	@task
	def consultation_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['consultation_analysis_task'],
			context=[self.conversation_generation_task],
			output_file='summary.txt'
		)

	@task
	def report_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config['report_generation_task'],
			context=[self.conversation_generation_task, self.consultation_analysis_task],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the DocGen crew"""
		print(self.tasks)
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)