set DOCKER_USER=qcooke
set DOCKER_VERSION=v0.3

echo %DOCKER_USER%
echo %DOCKER_VERSION%

echo authenticate
docker login

echo start build

cd MongoDB
echo build MongoDB
docker build -t %DOCKER_USER%/ts-mongodb:%DOCKER_VERSION% -f %~dp0\MongoDB/MongoDB.Dockerfile .
echo push MongoDB
docker push %DOCKER_USER%/ts-mongodb:%DOCKER_VERSION%
cd ..