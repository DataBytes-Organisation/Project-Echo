import random

def randompassword():
    string = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    length = 12
    newPassword = "".join(random.sample(string,length))
    return newPassword


def genotp():
    string = "0123456789"
    length = 6
    randOTP = "".join(random.sample(string,length))
    return randOTP