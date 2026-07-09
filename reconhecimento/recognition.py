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

from config import MODEL_NAME

def initialize_model():
    """
    Inicializa o modelo ArcFace (InsightFace).
    """

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"]
    )

    app.prepare(ctx_id=0)

    return app

#só chamar

# #app = initialize_model()

def recognize_face(app, database, image, threshold=0.60):
    """
    Recebe uma imagem contendo uma face
    e retorna nome e score.
    """

    faces = app.get(image)

    if len(faces) == 0:
        return None, 0

    embedding = faces[0].embedding

    best_name = None
    best_score = -1

    for name, db_embedding in database.items():

        score = cosine_similarity(
            embedding.reshape(1, -1),
            db_embedding.reshape(1, -1)
        )[0][0]

        if score > best_score:

            best_score = score
            best_name = name

    if best_score < threshold:

        best_name = "Desconhecido"

    return best_name, best_score

#só chamar 

#nome, score = recognize_face(
    #app,
    #database,
    #face_crop
#)