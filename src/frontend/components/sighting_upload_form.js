import React, { useState } from 'react';
import '../styles/sighting_upload_form.css';

const SightingUploadForm = () => {
  const [formData, setFormData] = useState({
    species: '',
    latitude: '',
    longitude: '',
    datetime: '',
    notes: '',
    file: null,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFileChange = (e) => {
    setFormData({ ...formData, file: e.target.files[0] });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    for (const key in formData) {
      data.append(key, formData[key]);
    }

    try {
      const response = await fetch('/api/upload_sightings', {
        method: 'POST',
        body: data,
      });
      if (response.ok) {
        alert('Sighting uploaded successfully!');
      } else {
        alert('Failed to upload sighting.');
      }
    } catch (error) {
      console.error('Error uploading sighting:', error);
    }
  };

  return (
    <form id="sighting-upload-form" onSubmit={handleSubmit}>
      <label htmlFor="species">Species Name:</label>
      <input
        type="text"
        id="species"
        name="species"
        value={formData.species}
        onChange={handleChange}
        required
      />

      <label htmlFor="latitude">Latitude:</label>
      <input
        type="number"
        id="latitude"
        name="latitude"
        min="-90"
        max="90"
        value={formData.latitude}
        onChange={handleChange}
        required
      />

      <label htmlFor="longitude">Longitude:</label>
      <input
        type="number"
        id="longitude"
        name="longitude"
        min="-180"
        max="180"
        value={formData.longitude}
        onChange={handleChange}
        required
      />

      <label htmlFor="datetime">Date and Time:</label>
      <input
        type="datetime-local"
        id="datetime"
        name="datetime"
        value={formData.datetime}
        onChange={handleChange}
        required
      />

      <label htmlFor="notes">Additional Notes:</label>
      <textarea
        id="notes"
        name="notes"
        rows="4"
        value={formData.notes}
        onChange={handleChange}
      ></textarea>

      <label htmlFor="file">Upload File (Photo/Video):</label>
      <input
        type="file"
        id="file"
        name="file"
        accept="image/*,video/*"
        onChange={handleFileChange}
      />

      <button type="submit">Submit</button>
    </form>
  );
};

export default SightingUploadForm;
