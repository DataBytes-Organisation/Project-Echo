set DOCKER_USER=qcooke
set DOCKER_VERSION=v0.3

echo %DOCKER_USER%
echo %DOCKER_VERSION%

echo authenticate
docker login

echo start build

cd HMI
echo build HMI
docker build -t %DOCKER_USER%/ts-hmi:%DOCKER_VERSION% -f %~dp0\HMI/HMI.Dockerfile .
echo push HMI
docker push %DOCKER_USER%/ts-hmi:%DOCKER_VERSION%
cd ..