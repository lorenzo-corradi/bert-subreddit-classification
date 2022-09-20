import streamlit as st
import config
import base64
import os.path
from app_backend import AppBackend
import matplotlib.pyplot as plt
import seaborn as sns

APP_TITLE = 'SubReddit predictor'
APP_SUBTITLE = 'This is an app helping you to choose the right SubReddit to post a submission in the field of data and analytics!'
BACKGROUND = os.path.join(config.IMAGES_DIR, 'univr_logo.png')

def run_backend(post_title, post_body, plot_tsne):
    try:
        with st.spinner('Please wait.'):
            app_backend = AppBackend()
            predictions_labels, predictions_confidence, tsne_to_plot = app_backend.run(
                post_title = post_title, 
                post_body = post_body, 
                plot_tsne = plot_tsne
            )
    except ValueError as value_error:
        st.error(value_error)
        return
        
    return predictions_labels, predictions_confidence, tsne_to_plot


def set_background(background):
    _, background_extension = os.path.splitext(background)
    
    background_extension = background_extension.lstrip('.')
    
    st.markdown(body = 
        f"""
        <style>
        .stApp{{
            background-image: url(data:image/{background_extension};base64,{base64.b64encode(open(background, "rb").read()).decode()});
            background-size: 15%;
            background-position: right top;
            background-repeat: no-repeat;
            background-origin: content-box;
            padding: 60px;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        """,
        unsafe_allow_html=True
    )
    
    return
    
if __name__ == '__main__':
    
    wide_layout = st.set_page_config(layout = 'wide')
    
    title = st.title(body = APP_TITLE)
    subtitle = st.markdown(body = 
        f"""
        <style>
        p {{
            font-size: 120%;
        }}
        </style>
        <p> {APP_SUBTITLE} </p>
        """,
        unsafe_allow_html=True
    )
    
    set_background(background=BACKGROUND)

    
    post_title = st.text_area(label = 'Write title of your Reddit submission.', height = 75)
    post_body = st.text_area(label = 'Write content of your Reddit submission.', height = 150)

    col_space_left, col_button, col_space_right = st.columns((4, 2, 4))

    with col_button:
        plot_tsne = st.checkbox(label = 'Plot TSNE')
        backend_button = st.button(label = 'Predict SubReddit')
        
        if (backend_button == True):
            predictions_labels, predictions_confidence, tsne_to_plot = run_backend(
                post_title = post_title, 
                post_body = post_body, 
                plot_tsne = plot_tsne
            )            
            
    col_space_left, col_result, col_space_right = st.columns((1, 5, 1))

    if (backend_button == True):
        with col_result:
            st.success(
                "You should probably post your submission in:\n\n" + 
                '\n'.join('{}: {}%\n'.format(*value) for value in zip(predictions_labels, predictions_confidence))
            )
            
        if (plot_tsne == True):
            with col_result:

                plt.figure(figsize = (10, 10))
                
                your_observation = tsne_to_plot.iloc[[-1]]
                your_observation['labels'] = 'your observation'
                
                sns.set(style='dark', color_codes=True)
                sns.scatterplot(
                    x='x', 
                    y='y', 
                    data=tsne_to_plot[:-1],
                    hue='labels', 
                    palette='bright', 
                    legend='full'
                )
                
                sns.scatterplot(
                    x='x', 
                    y='y', 
                    data=your_observation,
                    hue='labels',
                    palette=['black'],
                    legend='full',
                    s = 150
                    )

                # BUG: plt.figure breaks plot                
                st.pyplot()