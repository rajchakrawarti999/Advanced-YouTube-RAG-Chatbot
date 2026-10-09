"""
Step 7 - Evaluation.

- Ragas: scores a small QA set you supply against faithfulness, answer
  relevancy, context precision and context recall.
- LangSmith: needs no extra code here - once LANGCHAIN_API_KEY is set in
  .env (see config.py), every chain call in this app is traced
  automatically. Check https://smith.langchain.com after running a few
  questions through the app.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


def run_ragas_eval(questions, answers, contexts, ground_truths, llm, embeddings):
    """
    questions, answers, ground_truths : list[str]
    contexts                          : list[list[str]] (retrieved chunk
                                         texts for each question)
    Returns a pandas DataFrame with one row per question and a column per
    metric, ready to display in Streamlit.
    """
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()
