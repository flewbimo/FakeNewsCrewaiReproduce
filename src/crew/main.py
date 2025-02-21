
import sys
import os
from dotenv import load_dotenv

from crew import PoliticsliFakeNewsDetectionCrew
from crew import NonepoliticsFakeNewsDetectionCrew
from crew import RelatedToPoliticsCrew

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def run():
    inputs = {
        'article': "Did Kamala Harris Support Abortion Until the Time of Giving Birth?",
        'url': 'https://www.snopes.com/fact-check/kamala-harris-abortion-birth/'
    }
    IsPolitics = True if RelatedToPoliticsCrew.kickoff(inputs=inputs).raw=="True" else False
    if IsPolitics:
        print("是政治新闻")
        PoliticsliFakeNewsDetectionCrew.kickoff(inputs=inputs)
    else:
        print("不是政治新闻")
        NonepoliticsFakeNewsDetectionCrew.kickoff(inputs=inputs)

if __name__ == "__main__":
    run()