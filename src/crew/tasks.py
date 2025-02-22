
from crewai import Task
from textwrap import dedent
from agents import CustomAgents
from tools import AnalysisTools
article_analysis_tools = AnalysisTools()

class AnalysisTasks:
    def __init__(self):
        pass
    
    def IsPolitics(self):
        return Task(
            description=(
                """
                检查"({article})"是否是政治新闻.
                """
            ),
            expected_output="""
                仅仅回答"True" 或者 "False".
            """,
            agent=CustomAgents().Senior_Political_Expert(),
        )
    def Phrase(self):
        return Task(
            description=(
                """
                检查"({article})"是否存在耸人听闻的预告片、挑衅性或充满情感的语言或夸张的声明.
                """
            ),
            expected_output="""
                提供新闻是否存在耸人听闻的预告片、挑衅性或充满情感的语言或夸张的声明的简短总结.
            """,
            agent=CustomAgents().Senior_Linguistics_Expert(),
        )

    def Language(self):
        return Task(
            description=(
                """
                检查"({article})"中的语法错误、措辞错误、滥用引号或全部大写的单词.
                """
            ),
            expected_output="""
                提供新闻是否存在语法错误、措辞错误、滥用引号或全部大写的单词的简短总结.
            """,
            agent=CustomAgents().Senior_Grammar_Expert(),
        )

    def Commonsense(self):
        return Task(
            description=(
                """
                检查"({article})"新闻报道的合理性，并根据常识识别任何矛盾之处.
                """
            ),
            expected_output="""
                提供新闻的合理性与是否有矛盾之处的简短总结.
            """,
            agent=CustomAgents().Senior_Sociologist(),
        )

    def Standing(self):
        return Task(
            description=(
                """
                检查"({article})"是否促进了特定观点，而不是呈现客观事实.
                """
            ),
            expected_output="""
                提供新闻是否促进了特定观点，而不是呈现客观事实的简短总结.
            """,
            agent=CustomAgents().Senior_Political_Expert(),
        )

    def Search(self):
        return Task(
            description=(
                """
                搜索其他媒体关于"({article})"报告的任何冲突信息.
                """
            ),
            expected_output="""
                提供新闻冲突信息的简短总结.
            """,
            agent=CustomAgents().Senior_Search_Expert(),
            tools=[article_analysis_tools.search_tool(),
                   article_analysis_tools.scrape_tool()]
        )

    def URL(self):
        return Task(
            description=(
                """
                结合内部知识和外部数据库检查"({url})"是否来自缺乏可信度的域.外部数据库的名称是"url",表名是"myurl",包含网页的列为myurlcol.
                """
            ),
            expected_output="""
                提供url是否来自缺乏可信度的域的简短总结.
            """,
            agent=CustomAgents().Senior_Web_Page_Expert(),
            tools=[AnalysisTools().database_tool()],
        )

    def Summarization(self):
        return Task(
            description=(
                """
                结合已经得到的信息,得出"({article})"是真新闻还是假新闻.
                """
            ),
            expected_output="""
                首先说出该新闻是真新闻还是假新闻,仅仅回答"真新闻"或者"假新闻",然后再做出简短总结.
            """,
            agent=CustomAgents().Senior_Summarization_Expert(),
        )
    
   



