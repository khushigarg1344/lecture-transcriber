from transformers import pipeline

# Use the proper task and model
question_generator = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")

def generate_ai_flashcards(text, max_questions=5):
    if len(text.split()) > 500:
        text = " ".join(text.split()[:500])

    try:
        # Prepend prompt as required by the model
        input_text = "generate questions: " + text
        outputs = question_generator(input_text, max_length=128, do_sample=False, num_return_sequences=max_questions)
        
        # Basic post-processing: split Q/A (model often returns questions only, or question + answer)
        flashcards = []
        for out in outputs:
            qa = out['generated_text'].strip()
            if "?" in qa:
                q_part = qa.split("?")[0] + "?"
                a_part = qa.split("?")[1].strip()
                flashcards.append((q_part, a_part if a_part else "Answer not available."))
            else:
                flashcards.append((qa, "Answer not available."))
        
        return flashcards

    except Exception as e:
        print("Flashcard generation error:", e)
        return []

