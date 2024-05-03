from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define a Pydantic model for the object to be received
class ObjectToRegister(BaseModel):
    name: str
    value: int


# Define the function to run when an object is received
def process_object(received_object: ObjectToRegister):
    print("Received object:", received_object)
    # Here you can add more logic to process the received object
    return "Object processed successfully"


@app.post("/register_object")
async def register_object(obj: ObjectToRegister):
    # Process the received object
    result = process_object(obj)
    return {"message": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
