import streamlit as st
from streamlit_option_menu import option_menu
import pages as pg

st.set_page_config(layout="wide",
                   page_title="Twitter hijack detection",
                    page_icon="🐦")
st.markdown("""
<style>
.app-header {
    font-size:50px;
    color: #F63366;
    font-weight: 700;
}
.sidebar-header{
    font-family: "Lucida Sans Unicode", "Lucida Grande", sans-serif;
    font-size: 28px;
    letter-spacing: -1.2px;
    word-spacing: 2px;
    color: #FFFFFF;
    font-weight: 700;
    text-decoration: none;
    font-style: normal;
    font-variant: normal;
    text-transform: capitalize;
}
.positive {
    color: #000000;
    font-size:30px;
    font-weight: 700;  
}
.negative {
    color: #70F140;
    font-size:30px;
    font-weight: 700;  
}
</style>
""", unsafe_allow_html=True)

############################
#password implementation
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            
            # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        name = st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        name = st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

##############################################
##############################################
##############################################
#This will be the normal app you want to build
if check_password():
    
    ############################
    #title for the page
    
    st.write(f"""
        # Twitter hijack detection
            """)

    with st.sidebar:
        col1,col2 = st.columns([2,4])
        col1.image("gig pics/Black___White_Minimalist_Business_Logo__1_-removebg-preview.png",width=300,output_format='PNG')
        # original_title = '<p style="font-size: 40px;">Prediction</p>'
        # col2.markdown(original_title, unsafe_allow_html=True)
        selected = option_menu(
            menu_title = "",
            icons = ["hash","twitter","link","gear"],
            menu_icon = 'fire',
            options = ["Hashtag","Manual",'URL Entry'], # we can add settings bar in here list behind
            default_index = 0
        )
        
        # log_out = st.button('Log Out')
        # if log_out:
        #     import os
        #     os.system('streamlit cache clear')
    
    #this will load the particular pages from the pages.py        
    if selected == "Hashtag":
        pg.extract()
    if selected == "Manual":
        pg.predict()
    if selected =="URL Entry":
        pg.enter_url()
#    if selected =="Setting":
 #       pg.setting()