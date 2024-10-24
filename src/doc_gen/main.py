#!/usr/bin/env python
import sys
import time
import asyncio
import datetime
import random
from src.doc_gen.crew import DocGenCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding necessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

departments = [
    "Orofacial Pain and Oral Medicine",
    "Oral Pathology",
    "Oral and Maxillofacial Surgery",
    "Pediatric Dentistry",
    "Orthodontics",
    "Conservative Dentistry",
    "Prosthodontics",
    "Periodontics",
    "Advanced General Dentistry"
    ]
situations = [
    "a surgery", 
    "a medical consultation", 
    "discharge procedures", 
    "inpatient status check"
    ]

def get_today(date=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d")):
    return f"{int(date[:4])}/{int(date[4:-2]):02d}/{int(date[-2:]):02d}"

def log_execution_time(func):
    """
    Decorator that logs the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function <{func.__name__}> executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@log_execution_time
def run():
    """
    Run the crew.
    """
    inputs = {
        "department": random.choice(departments),
        "situation": random.choice(situations),
        "date": get_today(),
        "patient_name": "홍길동"
    }
    # asyncio.run(
    #     DocGenCrew().crew().kickoff_async(
    #         inputs=inputs))
    while True:
        DocGenCrew().crew().kickoff(inputs=inputs)

@log_execution_time
def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "department": random.choice(departments),
        "situation": random.choice(situations),
        "date": get_today(),
        "patient_name": "김덕배"
    }
    try:
        DocGenCrew().crew().train(
            n_iterations=int(sys.argv[1]), 
            filename=sys.argv[2], 
            inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

@log_execution_time
def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DocGenCrew().crew().replay(
            task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

@log_execution_time
def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "department": random.choice(departments),
        "situation": random.choice(situations),
        "date": get_today(),
        "patient_name": "덴컴 테스트 유저"
    }
    try:
        DocGenCrew().crew().test(
            n_iterations=int(sys.argv[1]), 
            openai_model_name=sys.argv[2], 
            inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
