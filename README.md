# Text-Summarizer

## To run the files

We used the Amazon Fine Food Reviews dataset and in order to run any of the notebooks you need to download it and get the "Reviews.csv" file. 

There are two parts of the project:

## RNN part

## Fine-tuning part

In this part, the notebooks were created and executed on the KTH cluster provided from the course. There are two notebooks, one for the training code _fine-tune-final.ipynb_ and the other for the demo/inference _fine-tune-demo.ipynb_.

Run _fine-tune-final.ipynb_:

-   To easily run it, first upload the file to the KTH cluser enviroment.

-   Create a folder called 'data' inside of the code directory and put the Reviews.csv file from the Amazon Fine Food Review dataset provided in the instruction document.

-   Run the notebook.

-   Three versions of the model will be saved as the model is training. The three latest versions of the model represent the model version from the previous 5000 iteration, 10 000 iterations and lastly 15 000 iterations. Once the training is done you find the models in the following path _./google-tf/tf-small_amazon/_. You will find three checkpoint folders.

Run _fine-tune-demo.ipynb_:

-   To easily run it, upload the file also to the KTH cluser enviroment in the same folder of the previous file.

-   Create a folder called 'final-model' and copy the content from the checkpoint folder you are interested in to this folder.

-   Run the notebook and now you will be able to do inference in the last code cell.

NOTE: If you don't want to run the notebooks on the Jupyter enviroment, then you will have to install the dependencies such datasets, transformers from HuggingFace.
