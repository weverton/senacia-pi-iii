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

from recognition import recognize_face


def evaluate_dataset(
        app,
        database,
        test_path,
        threshold=0.60
):
    """
    Percorre todas as imagens do conjunto
    de teste e calcula as métricas.
    """

    y_true = []
    y_pred = []

    # Percorre cada pasta (cada pessoa)

    for person in os.listdir(test_path):

        person_path = os.path.join(
            test_path,
            person
        )

        if not os.path.isdir(person_path):
            continue

        # Percorre todas as imagens daquela pessoa

        for image_name in os.listdir(person_path):

            image_path = os.path.join(
                person_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Aqui NÃO existe mais lógica
            # de reconhecimento.

            # Apenas chamamos a função
            # recognize_face()

            best_name, best_score = recognize_face(

                app,
                database,
                image,
                threshold

            )

            if best_name is None:
                continue

            y_true.append(person.strip())
            y_pred.append(best_name.strip())

            print(

                f"Real: {person:20}"

                f" Previsto: {best_name:20}"

                f" Score: {best_score:.3f}"

            )

    # -------------------------
    # Cálculo das métricas
    # -------------------------

    accuracy = sum(

        t == p

        for t, p in zip(y_true, y_pred)

    ) / len(y_true)

    print(f"Accuracy: {accuracy:.4f}")

    print(

        classification_report(

            y_true,

            y_pred,

            zero_division=0

        )

    )

    print(

        confusion_matrix(

            y_true,

            y_pred

        )

    )