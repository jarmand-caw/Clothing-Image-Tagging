# Clothing-Image-Tagging
Goal was to take a messy fashion dataset given by a company with (sometimes multiple) images per item and enrich the dataset with additional tag information. Accomplished with multiple ConvNets with similar architecture trained per tag type.

## Notebooks
- Notebooks labels "Genre","Body","Type","Color","Use", or "Chief"
  - Each contains all of the code neccesary to preprocess and transform the image data and subsequently train the net to predicct that tag
- "Data Intake" notebook was used to process the gross airtable csv export and download the images from the links extracted from the csv output
- "Format and visualize" takes the outputs of the net for each tag for each image and ads them to the csv file in the exact format it was given in so that it could be easily re-integrated into the company database

## Presentation
- Powerpoint file summarizing results and showing cool extras like being able to identify similar items of clothing by using the net outputs as an embedding
