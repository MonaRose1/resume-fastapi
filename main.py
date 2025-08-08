from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="AI Resume Generator")

# Load Hugging Face model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Define request format
class ResumeRequest(BaseModel):
    name: str
    skills: str
    experience: str
    job_title: str

@app.post("/generate-resume")
def generate_resume(data: ResumeRequest):
    prompt = (
        f"Create a professional resume for {data.name}, "
        f"applying for {data.job_title}. "
        f"Skills: {data.skills}. "
        f"Experience: {data.experience}. "
        f"Write it in formal bullet points."
    )

    output = generator(prompt, max_length=500, num_return_sequences=1)
    return {"resume": output[0]["generated_text"]}

@app.get("/")
def home():
    return {"message": "AI Resume Generator API is running!"}
