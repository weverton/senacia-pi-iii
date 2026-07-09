import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

def build_database(app, train_path, output_file):
    """
    Cria o banco de embeddings a partir das imagens de treino.
    """

    database = {}

    for person in os.listdir(train_path):

        person_path = os.path.join(train_path, person)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        print(f"Processando {person}")

        for image_name in os.listdir(person_path):

            image_path = os.path.join(
                person_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            faces = app.get(image)

            if len(faces) == 0:
                continue

            embeddings.append(
                faces[0].embedding
            )

        if len(embeddings) == 0:
            continue

        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        database[person] = mean_embedding

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print(f"Banco criado com {len(database)} pessoas.")

#só chamar

#build_database(
   # app,
   # TRAIN_PATH,
  #  OUTPUT_FILE
#)

def load_database(DATABASE_EMBEDDINGS_PATH):
    """
    Carrega o banco de embeddings.
    """

    with open(DATABASE_EMBEDDINGS_PATH, "rb") as f:

        database = pickle.load(f)

    return database

#database = load_database(
    #"/content/embeddings.pkl"
#)