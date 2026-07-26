@echo off
REM first build component images
docker-compose build

REM start the db container using env_db.txt which contains Database enviornment variables
docker run --rm -d --env-file .\env_db.txt -h db_server --name db_server_cont -p 27017:27017/tcp rb-echo-db:latest

REM get the ip of the database container and save it in a variable
echo database server is running on

FOR /F "tokens=*" %%i IN ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" db_server_cont') DO SET DB_IP=%%i
echo %DB_IP%

REM using the database container ip (variable above) and the env_api.txt to run the api container
docker run --rm -d --env-file .\env_api.txt -e DB_HOST=%DB_IP% -h api_server --name api_server_cont -p 9000:9000/tcp -p 9080:9080/tcp rb-echo-api:latest

REM get the ip of the api container and save it in a variable
FOR /F "tokens=*" %%i IN ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" api_server_cont') DO SET API_IP=%%i
echo API container up at %API_IP%

REM running the redis host and getting its ip
docker run --rm -d --name redis_server_cont -p 6379:6379/tcp redis:latest
FOR /F "tokens=*" %%i IN ('docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" redis_server_cont') DO SET REDIS_IP=%%i
echo REDIS container up at %REDIS_IP%

REM using redis ip, db ip and api ip, running the frontend! and using env_hmi.txt for other vars
docker run --rm -it --env-file .\env_hmi.txt -e DB_HOST=%DB_IP% -e REDIS_HOST=%REDIS_IP% -e API_HOST=%API_IP% --name web_server -p 8080:8080/tcp rb-echo-hmi:latest

pause