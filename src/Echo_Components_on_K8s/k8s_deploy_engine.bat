set DOCKER_USER=qcooke
set DOCKER_VERSION=v0.3

echo %DOCKER_USER%
echo %DOCKER_VERSION%

echo authenticate
docker login

echo start build

cd Engine
echo build engine
docker build -t %DOCKER_USER%/ts-engine:%DOCKER_VERSION% -f %~dp0\Engine/Engine.Dockerfile .
echo push engine
docker push %DOCKER_USER%/ts-engine:%DOCKER_VERSION%
cd ..