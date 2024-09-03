import {Bom} from 'node-bom'

const bom = new Bom({})



function getWeather(lat,long,time){
    bom.getForecastData(lat, long) // Sydney
  .then((data) => {
    console.log(data)
  })
}