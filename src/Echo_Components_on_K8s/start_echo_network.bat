@echo off
REM first build component images
docker-compose build

REM start the db container using env_db.txt which contains Database environment variables
docker-compose up -d db_server_cont
REM Wait for the db container to start
:wait_db
timeout /t 5 /nobreak >nul
docker inspect -f "{{.State.Running}}" db_server_cont | findstr "true"
if errorlevel 1 goto wait_db

REM get the IP of the database container
for /f "tokens=*" %%i in ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" db_server_cont') do set DB_IP=%%i
echo Database server is running on %DB_IP%

REM start the api container using env_api.txt
docker-compose up -d api_server_cont
REM Wait for the api container to start
:wait_api
timeout /t 5 /nobreak >nul
docker inspect -f "{{.State.Running}}" api_server_cont | findstr "true"
if errorlevel 1 goto wait_api

REM get the IP of the api container
for /f "tokens=*" %%i in ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" api_server_cont') do set API_IP=%%i
echo API container is up at %API_IP%

REM start the redis container
docker-compose up -d redis_server_cont
REM Wait for the redis container to start
:wait_redis
timeout /t 5 /nobreak >nul
docker inspect -f "{{.State.Running}}" redis_server_cont | findstr "true"
if errorlevel 1 goto wait_redis

REM get the IP of the redis container
for /f "tokens=*" %%i in ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" redis_server_cont') do set REDIS_IP=%%i
echo REDIS container is up at %REDIS_IP%

REM start the frontend container using env_hmi.txt
docker-compose up -d web_server
REM Wait for the frontend container to start
:wait_frontend
timeout /t 5 /nobreak >nul
docker inspect -f "{{.State.Running}}" web_server | findstr "true"
if errorlevel 1 goto wait_frontend

echo Frontend container is up and running.

pause



::I replaced docker run commands with docker-compose up commands, assuming you have a docker-compose.yml file defined with appropriate services.
::Each container is started with docker-compose up -d for detached mode, and then we wait for it to be running before proceeding.
::Error handling is implemented using labels (:wait_db, :wait_api, etc.) and checking the state of containers before proceeding.
::Reused variables for container IPs and improved output messages for better clarity.