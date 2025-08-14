from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import os
import subprocess

app = FastAPI()

# TEMP: Path to where your AnimateDiff models will live
ANIMATEDIFF_MODEL_DIR = "models/animatediff"
FRAMEPACK_MODEL_DIR = "models/framepack"

@app.post("/generate")
async def generate_video(
    prompt: str = Form(...),
    input_image: UploadFile = None
):
    # Save temp input image
    if input_image:
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, input_image.filename)
        with open(image_path, "wb") as f:
            f.write(await input_image.read())
    else:
        image_path = None

    # Output video path
    output_path = os.path.join(temp_dir, "output.mp4")

    # TODO: Replace with AnimateDiff + FramePack pipeline call
    # Example: subprocess call to your generation script
    subprocess.run([
        "python", "scripts/generate.py",
        "--prompt", prompt,
        "--input", image_path or "",
        "--output", output_path
    ])

    return FileResponse(output_path, media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
