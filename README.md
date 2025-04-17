For reason's I have not determined yet, you abosultely need to run this in a virtual environment.

Clone the repo (git clone https://github.com/DoctorRicko/cyb499_phishing_detection.git), cd to it, the run

python -m venv phishdetect # may take a minute
.\phishdetect\Scripts\activate

This should make the virtual environment and activate it, you should see (phishdetect) at the beginning of your CLI prompt.
Only NOW can you start installing things

pip install -r requirements.txt # I believe this downloads all necessary packages/dependencies/whatever

Assuming you have things set up correctly (I'm still determining datasets), this should correctly train our model,

python src/lora_train.py --model_name roberta-base --output_dir model/lora_test
