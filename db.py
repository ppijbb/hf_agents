from pymongo import MongoClient

def get_db_info(uri, db_name):
    client = MongoClient(uri)
    db = client[db_name]
    return [a for a in db.list_collections()]
    

# Example usage:
uri = "mongodb://127.0.0.1:27017/"
db_name = "mydatabase"
print(get_db_info(uri, db_name))
