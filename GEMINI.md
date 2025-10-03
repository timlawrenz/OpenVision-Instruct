remind the user at the start of a session to mount the data folder:
  sudo mount -t cifs //192.168.86.43/home/OpenGPT-4o-Image /home/tim/source/activity/OpenVision-Instruct/data/OpenGPT-4o-Image -o user=tim,uid=1000,gid=1000,file_mode=0664,dir_mode=0775
