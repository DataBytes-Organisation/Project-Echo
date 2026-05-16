import folium
from entities.entity import Entity
import os

class RenderedState(Entity):
    def __init__(self, lla=None) -> None:
        super().__init__(lla)

    def render_initial_sensor_state(self, config, animals):
        m = folium.Map(location=self.get_otways_coordinates()[0], zoom_start=13)
        for mic in config.SENSOR_MANAGER.MicrophoneObjects:
            folium.Marker(location=[mic.getLLA()[0], mic.getLLA()[1]], icon=folium.Icon(icon="microphone", prefix='fa', color="black"), custom_id=mic.unique_identifier).add_to(m)
        for animal in animals:
            folium.Marker(location=[animal.getLLA()[0], animal.getLLA()[1]], icon=folium.Icon(icon="dove", prefix="fa", color="green"), popup=str(str(animal.species) + '\n' + animal.uuid), custom_id=animal.uuid).add_to(m)

        m.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
        self.folium_map = m
    
    def render(self, animals):
        for animal in animals:
            for marker_ in self.folium_map._children.values():
                try:
                    if str(animal.uuid) == str(marker_.options['customId']):
                        marker_.location[0] = animal.getLLA()[0]
                        marker_.location[1] = animal.getLLA()[1]
                        marker_.icon.options['markerColor'] = 'green'
                except: pass

        self.folium_map.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
    
    def render_animal_vocalisation(self, animal):
        for marker_ in self.folium_map._children.values():
            try:
                if str(animal.uuid) == str(marker_.options['customId']):
                    marker_.location[0] = animal.getLLA()[0]
                    marker_.location[1] = animal.getLLA()[1]
                    marker_.icon.options['markerColor'] = 'red'
            except: pass

        self.folium_map.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'RenderedState.html'))
                