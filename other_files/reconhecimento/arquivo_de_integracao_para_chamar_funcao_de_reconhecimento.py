from recognition import initialize_model, recognize_face
from database import load_database
from config import DATABASE_PATH, THRESHOLD

# Inicializa o modelo 
app = initialize_model()

# Carrega o banco de embeddings 
database = load_database(DATABASE_PATH)

nome, score = recognize_face(
    app,
    database,
    crop
)

print(nome)
print(score)