import lmdb
import pickle

lmdb_path = "../data/odac/is2r/train"

env = lmdb.open(lmdb_path, readonly=True, lock=False)

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        data = pickle.loads(value)

        print("Ключ:", key)
        print("Значение:", data)
