from fastapi import FastAPI, APIRouter, UploadFile, File, Form, status, HTTPException
import os

app = FastAPI()
router = APIRouter()
UPLOAD_FOLDER = '/mnt/recordings/pre-processed'  # This should match the Docker volume

@router.post("/upload_wav", status_code=status.HTTP_201_CREATED)
async def upload_wav(file: UploadFile = File(...), new_file_name: str = Form(...)):
    print(f"Received file with content type: {file.content_type}")
    if file.content_type != 'audio/wave':
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav files are allowed.")
    
    if not new_file_name.endswith('.wav'):
        raise HTTPException(status_code=400, detail="New file name must have a .wav extension.")

    try:
        # Ensure the new file name is valid and doesn't contain any directory traversal characters
        new_file_name = os.path.basename(new_file_name)
        
        file_location = os.path.join(UPLOAD_FOLDER, new_file_name)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return {"message": "File uploaded successfully", "file_location": file_location}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))