from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
from agent import create_agent_from_csv
import tempfile

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the CSV Analysis Agent. Use /docs to interact."}

@app.post("/analyze/")
async def analyze_csv(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save uploaded file to temp path
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Create agent and run query
        agent = create_agent_from_csv(tmp_path)
        result = agent.run(question)

        return {"answer": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})