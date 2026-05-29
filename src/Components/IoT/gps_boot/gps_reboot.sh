#!/bin/bash
sudo systemctl stop gpsd.socket
sudo systemctl stop gpsd.service
sudo systemctl disable gpsd.socket
sudo systemctl disable gpsd.service
sudo killall gpsd
sudo systemctl enable gpsd.socket
sudo systemctl start gpsd.socket