import random
from crewai_tools import BaseTool
from src.doc_gen.tools.tool_data import DIALOGUE_SAMPLES

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

class MedicalDocumentTemplateTool(BaseTool):
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

class MedicalDialogueSampleTool(BaseTool):
	name: str = "MedicalDialogueSampleTool"
	description: str = (
		"Sample dialouge example for generation. Will give a random dialogue."
		)

	def _run(self, tool_input: str) -> str:
        # 다양한 보고서 유형에 따른 템플릿 반환
		return random.choice(DIALOGUE_SAMPLES).strip()
