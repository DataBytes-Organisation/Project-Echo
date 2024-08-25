CREATE DATABASE animal_data_db;

USE animal_data_db;

CREATE TABLE animals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    animal_name VARCHAR(255) NOT NULL,
    animal_type VARCHAR(255) NOT NULL,
    conservation_status VARCHAR(255) NOT NULL,
    audio_path VARCHAR(255),
    image_path VARCHAR(255) NOT NULL
);
