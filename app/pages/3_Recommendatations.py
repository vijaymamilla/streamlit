import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.header("Recommendation by user")

@st.cache_data
def dataframe_data():

    return pd.read_csv('app/artifactory/bl_tdidf_processed.csv')


@st.cache_data
def buildingModel():
    tf_idf = TfidfVectorizer(ngram_range=(2, 2), stop_words="english")

    tfidf_matrix = tf_idf.fit_transform(dataframe_data()['overview'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim
    #indices = pd.Series(dataframe_data().index, index=dataframe_data()['overview'])

def recommend(id):

    idx = dataframe_data()[dataframe_data()['id'] == id].index[0]

    cosine_sim = buildingModel()
    # Get the pairwise similarity scores between the input Property and all the properties
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the wines based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select the top three similarity scores
    sim_scores = sim_scores[1:10]

    # Get the property variety indices
    property_idx_list = [i[0] for i in sim_scores]

    df = dataframe_data()
    if len(property_idx_list) == 0:
        st.write("NO PROPERTIES FOUND")
    else:
        st.subheader("Top Matching PROPERTIES")
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

#dataframe_data()
#get_consine_similary()
recommend('bproperty-18')



