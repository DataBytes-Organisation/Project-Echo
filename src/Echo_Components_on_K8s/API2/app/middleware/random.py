import random

def randompassword():
    string = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    length = 12
    newPassword = "".join(random.sample(string,length))
    return newPassword