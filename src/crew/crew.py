
import ssl
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')



article_url = 'https://www.cnn.com/2024/11/05/us/tropical-storm-rafael-forecast-hurricane-hnk/index.html'

import requests
response = requests.get(article_url, verify=False)

from crewai import Crew, Process


from agents import CustomAgents
from tasks import AnalysisTasks




Phrase_Tool_agent = CustomAgents().Senior_Linguistics_Expert()
Language_tool_agent = CustomAgents().Senior_Grammar_Expert()
Commonsense_tool_agent = CustomAgents().Senior_Sociologist()
Standing_tool_agent = CustomAgents().Senior_Political_Expert()
Search_tool_agent = CustomAgents().Senior_Search_Expert()
URL_tool_agent = CustomAgents().Senior_Web_Page_Expert()
Summarization_agent = CustomAgents().Senior_Summarization_Expert()



IsPolitics_task = AnalysisTasks().IsPolitics()
Phrase_task = AnalysisTasks().Phrase()
Language_task = AnalysisTasks().Language()
Commonsense_task = AnalysisTasks().Commonsense()
Standing_task = AnalysisTasks().Standing()
Search_task = AnalysisTasks().Search()
URL_task = AnalysisTasks().URL()
Summarization_task = AnalysisTasks().Summarization()

RelatedToPoliticsCrew = Crew(
    agents=[
        Standing_tool_agent,
    ],
    tasks=[
        IsPolitics_task,
    ],
    process=Process.sequential,
    verbose=True,
)

PoliticsliFakeNewsDetectionCrew = Crew(
    agents=[
        URL_tool_agent,
        Phrase_Tool_agent,
        Language_tool_agent,
        Commonsense_tool_agent,
        Standing_tool_agent,
        Search_tool_agent,
        Summarization_agent,
        
        
    ],
    tasks=[
        URL_task,
        Phrase_task,
        Language_task,
        Commonsense_task,
        Standing_task,
        Search_task,
        Summarization_task,
        
    ],
    process=Process.sequential,
    verbose=True,
)

NonepoliticsFakeNewsDetectionCrew = Crew(
    agents=[
        URL_tool_agent,
        Phrase_Tool_agent,
        Language_tool_agent,
        Commonsense_tool_agent,
        Search_tool_agent,
        Summarization_agent,
        
    ],
    tasks=[
        URL_task,
        Phrase_task,
        Language_task,
        Commonsense_task,
        Search_task,
        Summarization_task,
    ],
    process=Process.sequential,
    verbose=True,
)
