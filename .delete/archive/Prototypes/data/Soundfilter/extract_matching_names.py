import pandas as pd

class BirdNameMatcher:
    def __init__(self, train_metadata_path, saved_bird_names_path, output_path):
        self.train_metadata_path = train_metadata_path
        self.saved_bird_names_path = saved_bird_names_path
        self.output_path = output_path

    # Load CSV data
    def load_data(self):
        self.saved_bird_names = pd.read_csv(self.saved_bird_names_path)
        self.train_metadata = pd.read_csv(self.train_metadata_path)

    # Find matching bird names
    def match_names(self):
        self.matched_birds = self.train_metadata[self.train_metadata['scientific_name'].isin(self.saved_bird_names['Scientific Name'])]
        self.matched_birds_unique = self.matched_birds.drop_duplicates(subset=['primary_label'])

    # Save the matched results to a CSV file
    def save_to_csv(self):
        self.matched_birds_unique[['scientific_name', 'primary_label']].to_csv(self.output_path, index=False)

    # Execute the matching process
    def run(self):
        self.load_data()
        self.match_names()
        self.save_to_csv()
        print(f"Successfully saved matched bird names to {self.output_path}")

# Paths to the files
train_metadata_path = 'train_metadata.csv'
saved_bird_names_path = 'Australian_bird_names.csv'
output_path = 'matched_bird_names.csv'

# Run the process
matcher = BirdNameMatcher(train_metadata_path, saved_bird_names_path, output_path)
matcher.run()
