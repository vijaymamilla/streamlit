import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from sentence_transformers import SentenceTransformer


st.header("Recommendations")

@st.cache_data
def dataframe_data():

    return pd.read_csv('app/artifactory/bl_tdidf_processed.csv')

@st.cache_data
def dataframe_data_transformer():
    return pd.read_csv('app/artifactory/bl_property_processed.csv')

def sbert_recommendation(query):

        embeddings_file = 'app/artifactory/property_recommend.npy'
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        query_embedding = model.encode([query])

        description_embeddings = np.load(embeddings_file)

        similarities = cosine_similarity(query_embedding, description_embeddings)

        top_n = 10
        top_n_indices = similarities[0].argsort()[-top_n:][::-1]

        df = dataframe_data_transformer()
        if len(top_n_indices) == 0:
            st.write("NO PROPERTIES FOUND")
        else:
            st.subheader("Top Matching PROPERTIES using SBERT")
            for index in top_n_indices:
                building_type = df.iloc[index]['building_type']
                building_nature = df.iloc[index]['building_nature']
                locality = df.iloc[index]['locality']
                price = df.iloc[index]['price']
                property_description = df.iloc[index]['property_description']
                image_url = df.iloc[index]['image_url']
                property_url = df.iloc[index]['property_url']

                st.image(image_url, caption="Property Image", width=100)
                st.markdown(f"**Property Type:** {building_type}")
                st.markdown(f"**Buiding Nature:** {building_nature}")
                st.markdown(f"**Location:** {locality}")
                st.markdown(f"**Price:** {price}")
                st.markdown(f"**Property Page:** [Link]({property_url})", unsafe_allow_html=True)
                st.markdown(f"**Description:** {property_description}")
                st.write("---")

@st.cache_data
def buildingModel():
    tf_idf = TfidfVectorizer(ngram_range=(2, 2), stop_words="english")

    tfidf_matrix = tf_idf.fit_transform(dataframe_data()['overview'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def recommend(id):

    idx = dataframe_data()[dataframe_data()['id'] == id].index[0]

    cosine_sim = buildingModel()
    # Get the pairwise similarity scores between the input Property and all the properties
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the properties based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top three similarity scores
    sim_scores = sim_scores[1:10]

    # Get the property variety indices
    property_idx_list = [i[0] for i in sim_scores]

    df = dataframe_data()
    if len(property_idx_list) == 0:
        st.write("NO PROPERTIES FOUND")
    else:
        st.subheader("Top Matching PROPERTIES using TDIDF")
        for index in property_idx_list:
            building_type = df.iloc[index]['building_type']
            building_nature = df.iloc[index]['building_nature']
            locality = df.iloc[index]['locality']
            price = df.iloc[index]['price']
            property_description = df.iloc[index]['property_description']
            image_url = df.iloc[index]['image_url']
            property_url = df.iloc[index]['property_url']

            st.image(image_url, caption="Property Image", width=100)
            st.markdown(f"**Property Type:** {building_type}")
            st.markdown(f"**Buiding Nature:** {building_nature}")
            st.markdown(f"**Location:** {locality}")
            st.markdown(f"**Price:** {price}")
            st.markdown(f"**Property Page:** [Link]({property_url})", unsafe_allow_html=True)
            st.markdown(f"**Description:** {property_description}")
            st.write("---")

property_dist = {'bproperty-16397':'Flat Can Be Found In Mohakhali For Sale, Near Mohakhali Dakkhin Para Jame Masjid','bproperty-12695': 'Business Space Is Up For Rent In The Most Convenient Location Of Banani Near Banani Bidyaniketan School & College','bproperty-8291': '8745 Sq Ft Commercial Space With Suitable Commercial Approaches For Rent In Double Mooring, Agrabad','bproperty-11703': '1950 Sq Ft Apartment For Rent In Gulshan 2 Near Ebl','bproperty-3108': 'Reside In This 2000 Square Feet Apartment For Rent In Mohammadpur, Iqbal Road','bproperty-7982': 'To Get A Trouble Free Life You Can Take Rent This 1200 Sq Ft House In Barontek','bproperty-12744': '1480 SFT Apartment For Sale In Extension Pallabi, Mirpur','bproperty-6573': 'Visit This Building For Sale In Adabor Near Maak Genius School','bproperty-15630': 'Don’t Waste Your Valuable Time! You Can Buy This 1650 Sq Ft Flat For Sale In Bochila','bproperty-13185': 'Reasonable 860 Sq Ft Under Constructed Flat Is Available For Sale In Mohammadpur Near To Dhaka Udyan Government College','bproperty-6116': 'Verify The Benefits Of This 2500 Sq Ft Office For Rent At South Agrabad Ward','bproperty-565': 'Buy This 2100 Square Feet Splendid Flat Available In Bashundhara R-a, Block I','bproperty-17196': 'Nice 1800 Sq Ft Home Is Available To Rent In Dhanmondi','bproperty-14922': 'Choose The Option Of Buying This 1300 Sq Ft Apartment In Mirpur 12','bproperty-8646': 'Residential Plot Is Available For Sale In Fatulla','bproperty-14182': '1672 Sq Ft Lovely Flat For Sale At 7 No. West Sholoshohor Ward','bproperty-3501': 'Buy This 1250 Square Feet Flat Available In Bashundhara R-a','bproperty-6991': 'Nicely Planned Flat Of 1888 Sq Ft In Bashundhara R-a For Rent Nearby North South University','bproperty-2683': 'A well-constructed 1100 SQ FT flat is for sale in Mirpur, Block A','bproperty-11348': '1500 Sq Ft Residential Apartment For Sale In Eastern Pallabi, Mirpur'}

r_type = st.radio("Select Recommendation Type",   ( 'SBERT','TFIDF'))

option = st.selectbox("What\'s your favorite property",list(property_dist.keys()))

if option and r_type:

    if r_type == 'TFIDF':
        recommend(option)
    else:
        sbert_recommendation(property_dist[option])

