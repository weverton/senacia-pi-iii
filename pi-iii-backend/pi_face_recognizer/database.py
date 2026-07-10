import pickle

def load_database(database_path):
    """
    Carrega o banco de embeddings.
    """

    with open(database_path, "rb") as f:
        database = pickle.load(f)

    return database