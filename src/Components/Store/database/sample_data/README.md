## sample data for populating database and testing dataflow between components

1. events.json is the data from the simlulator to the engine, containing timestamp, location, sensor and audio data.
2. eventsClassified.json is the data from the engine to the database. It contains all the fields in events.json plus species and detection confidence
3. movements.json is the data from simulator to the databse, and comtains true animal locations
4. species.json contains information about each species.