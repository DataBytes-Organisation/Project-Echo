
from myfunctions import myFunctions

# Set the global parameters
filepath = r"D:\SIT374-Gourp Project A\Test dataset\Test dataset\Acanthiza chrysorrhoa\region_3.650-4.900.MP3"
input_path = r"D:\SIT374-Gourp Project A\Test dataset\Test dataset\Acanthiza chrysorrhoa"
output_path = r"D:\SIT374-Gourp Project A\Test dataset\Augmented Data\test.mp3"
plt_x_axis = True
plt_y_axis = True
plt_title = True
image_title = "Audio Waveform"
image_width = 10
image_height = 10
number_of_images_to_display = 2
augmentation = "AddGaussianNoise"
min_value = 0.001
max_value = 0.015

# Calling the functions from myfunction.py
testFunctions = myFunctions()
testFunctions.DisplayWave(filepath, plt_x_axis, plt_y_axis, plt_title, image_title, image_width, image_height)
testFunctions.show_example_images(input_path, number_of_images_to_display)
testFunctions.display_img_after_aug(filepath, output_path)
testFunctions.apply_augmentation(filepath, output_path, augmentation, min_value, max_value)
