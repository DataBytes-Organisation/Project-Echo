set DOCKER_USER=qcooke
set DOCKER_VERSION=v0.3

echo %DOCKER_USER%
echo %DOCKER_VERSION%

echo authenticate
docker login

echo start build

cd API
echo build API
docker build -t %DOCKER_USER%/ts-api:%DOCKER_VERSION% -f %~dp0\API/API.Dockerfile .
echo push API
docker push %DOCKER_USER%/ts-api:%DOCKER_VERSION%
cd..