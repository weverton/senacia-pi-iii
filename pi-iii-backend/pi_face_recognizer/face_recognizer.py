from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, model_path, database):
        """
        Inicializa o modelo ArcFace (InsightFace).
        """

        self.app = FaceAnalysis(
            name=model_path,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0)
        
        self.database = database


    def recognize_face(self, image, threshold=0.60):
        """
        Recebe uma imagem contendo uma face
        e retorna nome e score.
        """

        faces = self.app.get(image)

        if len(faces) == 0:
            return None, 0

        embedding = faces[0].embedding

        best_name = None
        best_score = -1

        for name, db_embedding in self.database.items():
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