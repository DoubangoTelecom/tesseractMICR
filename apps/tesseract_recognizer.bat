REM Usage: tesseract_recognizer.exe path_to_images_folder path_to_tessdata_folder
REM path_to_images_folder -> relative or absolute path to folder containing the images to process
REM path_to_tessdata_folder -> relative or absolute path to folder containing *.traineddata files
REM example: tesseract_recognizer.exe ./images ../tessdata_fast
REM another example: tesseract_recognizer.exe ./images ../tessdata_best

tesseract_recognizer.exe ./images ../tessdata_fast