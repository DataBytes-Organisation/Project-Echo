# data/

Small, git-trackable sample datasets and test fixtures used across teams. Large
model weights, mel-spectrogram caches, and other multi-gigabyte local artifacts
stay under their existing (gitignored) locations rather than here.

- `weather/` — sample BOM weather station CSV used by the API weather-lookup routes.
- `samples/store_audio/` — sample animal call `.wav` files used to seed the Store/Mongo database.
- `test_fixtures/` — small audio files used for manual/API testing (`pig.wav`, `decoded.wav`, `sample_uploads/`).
