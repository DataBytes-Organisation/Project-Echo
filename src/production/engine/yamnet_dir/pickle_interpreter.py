import pickle


"your_file.pkl"
def read_pickle(file):
    # Open the .pkl file in binary read mode
    with open(file, "rb") as pkl_file:
        # Load the Pickle object from the file
        loaded_object = pickle.load(pkl_file)

        # Now, you can inspect the loaded object
        print(loaded_object)

read_pickle("class_names.pkl")
read_pickle("label_encoder.pkl")
