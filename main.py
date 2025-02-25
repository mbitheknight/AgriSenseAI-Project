# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
<<<<<<< HEAD
    return {"message": "Hello, World!"}
=======
    return {"message": "Hello, World!"}
>>>>>>> 8bc61519f539220db9aa26125010bcb1799274e0
