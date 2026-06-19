from flask import Flask, render_template, request
import os
import whisper
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import yt_dlp as youtube_dl
import re
nltk.download("stopwords")
from nltk.corpus import stopwords

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering")

# Extract keywords
def extract_keywords(text, top_n=10):
    stop_words = stopwords.words("english")
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=top_n)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    return list(keywords)

# Summarize transcript
def summarize_text(text):
    word_count = len(text.split())
    if word_count < 30:
        return "Text too short to summarize."
    if word_count > 500:
        text = " ".join(text.split()[:500])
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Generate flashcards using QA pipeline
def generate_flashcards(transcript):
    questions = []
    answers = []

    sentences = [s.strip() for s in transcript.split(".") if len(s.split()) > 5]
    for sent in sentences[:5]:
        question = f"What is explained in: \"{sent}\"?"
        try:
            result = qa_pipeline(question=question, context=transcript)
            questions.append(question)
            answers.append(result["answer"])
        except:
            continue

    return list(zip(questions, answers))

# Transcribe audio file
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    transcript = result["text"].strip()
    if not transcript:
        raise ValueError("Empty transcript generated.")
    keywords = extract_keywords(transcript)
    summary = summarize_text(transcript)
    flashcards = generate_flashcards(transcript)
    return transcript, keywords, summary, flashcards

# Download and convert YouTube to audio
def download_youtube_audio(url, output_dir="uploads"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/yt_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return os.path.splitext(filename)[0] + ".mp3"

# Homepage route
@app.route("/")
def home():
    return render_template("home.html")

# Transcription route
@app.route("/transcribe", methods=["GET", "POST"])
def transcribe():
    transcript = ""
    keywords = []
    summary = ""
    flashcards = []
    error = None

    if request.method == "POST":
        try:
            youtube_url = request.form.get("youtube_url")
            if youtube_url:
                file_path = download_youtube_audio(youtube_url)
                transcript, keywords, summary, flashcards = transcribe_audio(file_path)

            elif "file" in request.files and request.files["file"].filename != "":
                file = request.files["file"]
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                transcript, keywords, summary, flashcards = transcribe_audio(file_path)
            else:
                error = "Please upload a file or paste a YouTube link."
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary,
        keywords=keywords,
        flashcards=flashcards,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
