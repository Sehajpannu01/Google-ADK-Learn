from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from tests.conftest import EVAL_MODEL
from greeting_agent.agent import root_agent
from google.adk.sessions import AgentSession


# def test_greeting_variations():
    # test_cases = [
    #     LLMTestCase(
    #         input="hi",
    #         actual_output="Hi there! What’s your name?",
    #         expected_output="Friendly greeting asking for name",
    #         retrieval_context=["User is asking for a greeting"]
    #     ),
    #     LLMTestCase(
    #         input="hey",
    #         actual_output="Hey! May I know your name?",
    #         expected_output="Casual greeting with name request",
    #         retrieval_context=["User is asking for a greeting"]
    #     ),
    #     LLMTestCase(
    #         input="good morning",
    #         actual_output="Good morning! What’s your name?",
    #         expected_output="Polite greeting with personalization",
    #         retrieval_context=["User is asking for a greeting"]
    #     ),
    #     LLMTestCase(
    #     input="what is your price?",
    #     actual_output="Hello! May I know your name?",
    #     expected_output="Agent should not hallucinate and should redirect politely",
    #     retrieval_context=["User intent is unclear or not a greeting"]
    #     ),
    #     LLMTestCase(
    #     input="??",
    #     actual_output="Hello! May I know your name?",
    #     expected_output="Agent should not hallucinate and should redirect politely",
    #     retrieval_context=["User intent is unclear or not a greeting"]
    #     ),
# ]

def test_greeting_variations():
    session = AgentSession(agent=root_agent)

    inputs = [
        ("hi", "Friendly greeting asking for name", "User is asking for a greeting"),
        ("hey", "Casual greeting with name request", "User is asking for a greeting"),
        ("good morning", "Polite greeting with personalization", "User is asking for a greeting"),
        ("what is your price?", "Agent should not hallucinate and should redirect politely", "User intent is unclear or not a greeting"),
        ("??", "Agent should not hallucinate and should redirect politely", "User intent is unclear or not a greeting"),
    ]

    test_cases = []

    for user_input, expected, context in inputs:
        actual_output = session.send(user_input)

        test_cases.append(
            LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                expected_output=expected,
                retrieval_context=[context]
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL),
        ]
    )

    # answer_relevancy_metrics= AnswerRelevancyMetric(model=EVAL_MODEL, threshold=0.95)
    # evaluate(
    #     test_cases=test_cases,
    #     metrics=[
    #         answer_relevancy_metrics,
           
    #     ]
    # )

