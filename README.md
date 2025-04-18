For reason's I have not determined yet, you abosultely need to run this in a virtual environment.

# Virtual Environment
```
git clone https://github.com/DoctorRicko/cyb499_phishing_detection.git
```
cd to it, then run
```
python -m venv phishdetect # may take a minute
.\phishdetect\Scripts\activate
```
This should make the virtual environment and activate it, you should see (phishdetect) at the beginning of your CLI prompt.

# Configure Virtual Environment
Only NOW can you start installing things. YOU WILL DO THE REST OF THE WORK IN THIS ENVIRONMENT
```
pip install -r requirements.txt # I believe this downloads all necessary packages/dependencies/whatever
```
# Acquiring datasets
You need to manually get the datasets and put them in your LOCAL file structure (you cannot put them on github)
```
mkdir -p data/extracted/phishing data/extracted/enron
```
You need to make this *exact* file path at the top of the folder (/cyb499_phishing_detection)
Then we need to download the csv files and *manually* place the zip/tar.gz files in the approriate folders
You *may* need to download 7zip and adjust environment variables: https://www.7-zip.org/

PHISHING EMAILS DATASET:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?resource=download&select=Enron.csv
-----> data/extracted/phishing
```
7z x data/raw/enron.csv.zip -odata/extracted/phishing
```
LEGITIMATE EMAILS DATASET:
https://www.cs.cmu.edu/~enron/ (May 7, 2015 Version of dataset hyperlink) ------> data/extracted/enron
```
7z x data/extracted/enron/enron_mail_20150507.tar -odata/extracted/enron
```
run the 7z command at the top of the folder, or adjust your pathing accordingly.

# Training the Model
Assuming you have things set up correctly (I'm still determining datasets), this should correctly train our model,

IMPORTANT: Training is a heavy load on your computer and will take *hours* to complete. 
```
python src/lora_train.py --model_name roberta-base --output_dir model/lora_test
```
# Evaluating the Model
Now that you have the model, all that's left is simply to put it to the test. This doesn't take as long since
I reduced the amount of files it needs to go over, if you change the load you could overload your CPU and crash
the program, maybe even your computer

```
python .\src\evaluate.py --model_path .\model\lora_test\ --test_file .\data\test.csv --results_dir .\results
```

I believe that's everything, you can change evaulation.py to give you different metrics if you'd like, but
you'd probably have to modify the rest of the code as well.
