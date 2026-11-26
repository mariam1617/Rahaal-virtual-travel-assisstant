"""Quick manual test:  python -m src.rag "Is alcohol allowed?"   (works offline)"""
import sys

from src.rag.answer import answer_tourist_query

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = answer_tourist_query(question)
    print(f"[{result.status}]\n{result.answer}")
    if result.pending_question:
        print(f"\n(pending question: {result.pending_question})")
