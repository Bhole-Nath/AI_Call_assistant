#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AiCallbotCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    inputs = {
        'topic': 'To act as a sales executive with 15+ years experience in Telesales. You specializes in understanding the needs of the business and proposing our AI Assistant call support Software services for their business.',
        'current_year': str(datetime.now().year)}
    try:
        AiCallbotCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "Perfect ways, motivation and vocabulary to use while on call conversation for getting job done",
        'current_year': str(datetime.now().year)
    }
    try:
        AiCallbotCrew().crew().train(n_iterations=int(sys.argv[2]), filename=sys.argv[3], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AiCallbotCrew().crew().replay(task_id=sys.argv[2])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI uses in handling call conversation",
        "current_year": str(datetime.now().year)
    }
    
    try:
        AiCallbotCrew().crew().test(n_iterations=int(sys.argv[2]), eval_llm=sys.argv[3], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    out = globals()[cmd]()
    if out is not None:
        print(out)
