# TwitterMonitoring

Description:

A Sentiment Analysis Application which is able to classify Tweets to "postive", "neutral" or "negative" class.

Run:

1. Set an environment variable for `FLASK_APP`. On Linux and macOS, use `export set FLASK_APP=webapp`; on Windows use `set FLASK_APP=webapp`

2. Navigate into the backend folder, then launch the program using `python -m flask run`

3. Navigate into the frontend folder, then launch the program using `python app.py`

How to use:

1. Enter search keywordsand click "run"

![image](https://user-images.githubusercontent.com/56144156/107477095-21c76b00-6bb2-11eb-88ca-05dd07685017.png)

2. A dashboard will show the numbers of Tweets predicted to corresponding classes at different times. Also, there is a table at the bottom of the web page that displays the Tweets collected from Twitter.

![image](https://user-images.githubusercontent.com/56144156/107477105-268c1f00-6bb2-11eb-8158-a8bde3a59845.png)


Overview:

Fronend: Python Dash

Backend: Python Flask

Text Clssification Model: CNN on GloVe word embedding
