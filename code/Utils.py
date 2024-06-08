import os
import pickle


def create_paths(list_paths):
    for path in list_paths:
        if not os.path.exists(path):
            os.makedirs(path)


def find_intersection(list1, list2):
    mutuals = list(set(list1).intersection(list2))
    return mutuals


def pickle_data(data_to_store, path):
    with open(path, 'wb') as file_pi:
        pickle.dump(data_to_store, file_pi)


def read_pickled_data(path):
    pickle_in = open(path, 'rb')
    data = pickle.load(pickle_in)
    return data
