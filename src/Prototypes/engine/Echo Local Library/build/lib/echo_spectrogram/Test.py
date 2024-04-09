from myfunctions import AUDIO_FMAX
from myfunctions import myFunctions

# Set the global parameters
filepath = r"Your file path here"
input_path = r"Your folder path here"
output_path = r"Output folder path here"
plt_x_axis = True
plt_y_axis = True
plt_title = True
image_title = "Audio Waveform"
image_width = 20
image_height = 10
number_of_images_to_display = 2
augmentation = "AddGaussianNoise"
min_value = 0.001
max_value = 0.015

# Calling the functions from myfunction.py
print(AUDIO_FMAX)
testFunctions = myFunctions()
testFunctions.DisplayWave(filepath, plt_x_axis, plt_y_axis, plt_title, image_title, image_width, image_height)
testFunctions.show_example_images(input_path, number_of_images_to_display)
testFunctions.display_img_after_aug(filepath, output_path)
testFunctions.apply_augmentation(filepath, output_path, augmentation, min_value, max_value)
