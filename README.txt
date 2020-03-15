Hi readers!


To run my program, type in the following command in the terminal:
	
	main.py <is_jpg> <with_edge> <crop>

Where  <is_jpg> <with_edge> <crop> are three arguments for the main file which are all boolean values.

<is_jpg>: can either be "true" or "false". If "true", the program is going to process all the images that are jpg files. If "false", the program is going to process all images that are tif files.

<with_edge>: whether you want to use edge detection to align images or not. If "true", the  program will run alignment algorithm with filtered images. If "false", the alignment algorithm will only compare the raw pixel value to calculate SSD.

<crop>: whether you want to crop out the borders of the output images or not. If "true", it will crop, if false, it will not crop.



----------------------------------------------------------
Note1: "true" and "false" has to be in lower case eitherwise it will not work!

Note2: the image you want to process(input images) should be put into a folder called "dataset" in the same directory as the main.py file. This is because I set the file path to load image as "./dataset/*".

Note3: output images will be put into a folder called "result", which will appear in the same directory as main.py.



----------------------------------------------------------

Example

Typing "main.py false true true" in the terminal will align all images that are .tif file using edge detection. The output images will have their borders cropped.