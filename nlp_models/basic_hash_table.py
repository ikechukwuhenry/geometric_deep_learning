import numpy as np



def basic_hash_table(values, num_buckets):
    """
    A simple hash table implementation that maps input values to buckets.

    Args:
        values (list): A list of input values to be hashed.
        num_buckets (int): The number of buckets in the hash table.

    Returns:
        dict: A dictionary representing the hash table with bucket indices as keys
              and lists of values as values.
    """
    def hash_function(value, num_buckets):
        return int(value) % num_buckets
    
    hash_table = {i: [] for i in range(num_buckets)}
    for value in values:
        hash_value = hash_function(value, num_buckets)
        hash_table[hash_value].append(value)
    return hash_table

# gettig the direction of a point v with respect to a plane P
def side_side_of_plane(P, v):
    dot_product = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dot_product)
    return sign_of_dot_product

# locality(i.e location ) sensitive hashing function.
def hash_multiple_plane(P_l, v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_side_of_plane(P, v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i
    return hash_value


def side_of_plane_matrix(P, v):
    dot_product = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dot_product)
    return sign_of_dot_product
  



# Example usage:
if __name__ == "__main__":
    values = [10, 22, 31, 4, 15, 28, 17, 88, 59]
    num_buckets = 7
    hash_table = basic_hash_table(values, num_buckets)
    print("Hash Table:")
    for bucket, vals in hash_table.items():
        print(f"Bucket {bucket}: {vals}")


    num_dimensions = 2 # 300 in assignment
    num_planes = 3   # 10 in assignment
    random_planes_matrix = np.random.normal(size=(num_planes, num_dimensions))

    v = np.array([[2,1]])
    num_planes_matrix = side_of_plane_matrix(random_planes_matrix, v)
    print(f"Side of plane matrix:\n{num_planes_matrix}")


    # Docuement vectors
    word_embedding = {
        "I": np.array([1, 0, 1]),
        "love": np.array([-1, 0, 1]),
        "learning": np.array([1, 0, 1]),
    }

    words_in_document = ["I", "love", "learning"]
    document_embedding = np.array([0, 0, 0])

    for word in words_in_document:
        document_embedding += word_embedding.get(word, 0)

    print(f"Document embedding: {document_embedding}")
