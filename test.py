from backend.app.generation.answer import answer_question

print("Sending question to pipeline...")
result = answer_question("What library does the T5 team leverage for ease of implementation of model and data parallelism?")

print("\n ANSWER")
print(result["answer"])