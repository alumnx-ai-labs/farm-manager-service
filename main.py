from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hi"}

@app.get("/hello")
def read_hello():
    return {"message": "Hello World"}
