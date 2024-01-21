set DOCKER_USER=qcooke
set DOCKER_VERSION=v0.3

echo %DOCKER_USER%
echo %DOCKER_VERSION%

echo authenticate
docker login

echo start build

cd Simulator
echo build simulator
docker build -t %DOCKER_USER%/ts-simulator:%DOCKER_VERSION% -f %~dp0\Simulator/Simulator.Dockerfile .
echo push simulator
docker push %DOCKER_USER%/ts-simulator:%DOCKER_VERSION%
cd ..

cd Engine
echo build engine
docker build -t %DOCKER_USER%/ts-engine:%DOCKER_VERSION% -f %~dp0\Engine/Engine.Dockerfile .
echo push engine
docker push %DOCKER_USER%/ts-engine:%DOCKER_VERSION%
cd ..

cd API
echo build API
docker build -t %DOCKER_USER%/ts-api:%DOCKER_VERSION% -f %~dp0\API/API.Dockerfile .
echo push API
docker push %DOCKER_USER%/ts-api:%DOCKER_VERSION%
cd..

cd HMI
echo build HMI
docker build -t %DOCKER_USER%/ts-hmi:%DOCKER_VERSION% -f %~dp0\HMI/HMI.Dockerfile .
echo push HMI
docker push %DOCKER_USER%/ts-hmi:%DOCKER_VERSION%
cd ..

cd MongoDB
echo build MongoDB
docker build -t %DOCKER_USER%/ts-mongodb:%DOCKER_VERSION% -f %~dp0\MongoDB/MongoDB.Dockerfile .
echo push MongoDB
docker push %DOCKER_USER%/ts-mongodb:%DOCKER_VERSION%
cd ..

cd MQTT
echo build MQTT
docker build -t %DOCKER_USER%/ts-mqtt:%DOCKER_VERSION% -f %~dp0\MQTT/MQTT.Dockerfile .
echo push MQTT
docker push %DOCKER_USER%/ts-mqtt:%DOCKER_VERSION%
cd ..