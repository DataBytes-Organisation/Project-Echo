The display audio images file includes two functions that can be used to display an image of the 
audio clip provided.

The functions are as follows

DisplayWave(filepath,plt_x_axis,plt_y_axis,plt_title,image_title):
This function displays the audio as a wave function and incluides a number of optional values that can be included 
on the image

Variables:
filepath: String, path of the audio file must be provided
plt_x_axis: Boolean value, entering true plots values on the x axis, entering False removes the legend from the x axis
plt_y_axis:Boolean value, entering true plots values on the y axis, entering False removes the legend from the y axis
plt_title: Boolean value, entering true plots a title at the top of the image, entering False removes the title from the image
image_title: String: If plt_title is true, this string will be presented as a title at the top of the image



DisplayMelSpec(filepath,plt_x_axis,plt_y_axis,plt_title,plt_colourBar,image_title):
This function displays the audio converted into a mel-spectrogram and incluides a number of optional values that can be included 
on the image

Variables:
filepath: String, path of the audio file must be provided
plt_x_axis: Boolean value, entering true plots values on the x axis, entering False removes the legend from the x axis
plt_y_axis:Boolean value, entering true plots values on the y axis, entering False removes the legend from the y axis
plt_title: Boolean value, entering true plots a title at the top of the image, entering False removes the title from the image
plt_colourBar:Boolean value, entering true plots a colour bar legend on the side of the image to show the values of the colours in the image
image_title: String: If plt_title is true, this string will be presented as a title at the top of the image