from recognition import initialize_model
from database import build_database
from database import load_database
from evaluation import evaluate_dataset

from config import (
    TRAIN_PATH,
    TEST_PATH,
    DATABASE_EMBEDDINGS_PATH,
    THRESHOLD
)

def main():

    app = initialize_model()

    build_database(
        app,
        TRAIN_PATH,
        DATABASE_EMBEDDINGS_PATH
    )

    database = load_database(
        DATABASE_EMBEDDINGS_PATH
    )

    evaluate_dataset(
        app,
        database,
        TEST_PATH,
        THRESHOLD
    )

if __name__ == "__main__":
    main()