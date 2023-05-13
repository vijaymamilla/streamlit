# streamlit
Streamlit Project with ML and Docker Integration

Run the following commands to run locally

pip install -r requirements.txt

streamlit run app/Home.py

Docker commands -

docker build -t appimage .

docker run -d --name appcontainer -p 8501:8501 appimage

