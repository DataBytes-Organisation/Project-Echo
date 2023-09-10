## Kaggle Birdclef 2023
## Finding Australian Birds in the birdclef database 

The train_medata.csv is the information of all the bird sounds in the birdclef database the following dataset contains various aspects such as:
1. primary_label - The label of the sound files.
2. secondary_label - A secondary label of the sound files.
3. type - The type of the sound 
4. latitude - The latitude of where the bird sound was collected.
5. longitude - The longitude of where the bird sound was collected.
6. scientific_name - The scientific name of the bird species. 
7. common_name - The common name of the bird species.
8. author - the author for the file.
9. licence - the licence for the audio file. 
10. rating - the rating for the file.
11. url - the url for the sound. 
12. filename - the name of the audio file. 

To further process this data and to find if the whole dataset contains any Australian birds we can use the latitude and longitudes to pinpoint Australian locations. 

Australia is roughly bounded by the following latitudinal and longitudinal coordinates:

Latitude: -10 to -44
Longitude: 113 to 154

Found 20 entries within the dataset containing sounds of birds such as Black Kite, Striated Heron, Laughing Dove, Little Egret and many more and various types such as flight call, call, alarm call, begging call etc.

