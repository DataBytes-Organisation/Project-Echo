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