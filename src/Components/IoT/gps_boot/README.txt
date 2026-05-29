To install and enable the automatic GPS reboot fix please 
follow the provided steps:

1. Copy the gps_reboot.sh file to the Raspberry Pi at:
    /home/pi/

2. Copy the gps-reboot.service file to the Raspberry Pi at:
    /etc/systemd/system/

3. Open the terminal

4. Reload daemon by running: 
    sudo systemctl daemon-reload

5. Start the service by running: 
    sudo systemctl start gps-reboot.service

6. Enable the service to run at boot: 
    sudo systemctl enable gps-reboot.service
    
7. Restart Raspberry Pi