import logging
import streamlit as st
import openai
from pinecone import Pinecone
from datetime import datetime
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

#------------------- Utility Functions -------------------
def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if "query" not in st.session_state:
        st.session_state.query = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "retrieved_texts" not in st.session_state:
        st.session_state.retrieved_texts = None
    if "retrieved_pdf_title" not in st.session_state:
        st.session_state.retrieved_pdf_title = None
    if "retrieved_pdf_page" not in st.session_state:
        st.session_state.retrieved_pdf_page = None
    if "retrieved_pdf_link" not in st.session_state:
        st.session_state.retrieved_pdf_link = None
    if "query_input" not in st.session_state:
        st.session_state.query_input = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None  # Initialize feedback state with a default value
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False  # Track feedback submission status


def clear_text_area():
    """
    Clear the text area and reset session state variables.
    """
    st.session_state.submit_clicked = False
    st.session_state.query_input = None
    st.session_state.query= None
    st.session_state.response = None
    st.session_state.retrieved_texts =None
    st.session_state.feedback = None  # Reset feedback state
    st.session_state.feedback_submitted = False
    st.session_state.retrieved_pdf_title = None
    st.session_state.retrieved_pdf_page = None
    st.session_state.retrieved_pdf_link = None  # Reset feedback submission status

def submit_text():
    st.session_state.submit_clicked = False
    st.session_state.retrieved_pdf_title = None
    st.session_state.retrieved_pdf_page = None
    st.session_state.retrieved_pdf_link = None
    st.session_state.response = None
    st.session_state.retrieved_texts =None
    st.session_state.feedback = None  # Reset feedback state
    st.session_state.feedback_submitted = False

#-------------------- Drive -----------------
# Authenticate and connect to Google Sheets
# def connect_to_google_sheet(sheet_name):
#     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#     credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
#     client = gspread.authorize(credentials)
#     sheet = client.open(sheet_name).sheet1
#     return sheet
def connect_to_google_sheet(sheet_name):
    """
    Connect to Google Sheets using credentials stored in Streamlit secrets.
    """
    try:
        # Load credentials from Streamlit secrets
        credentials_dict = json.loads(st.secrets["google_sheets"]["credentials"])
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            credentials_dict,
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(credentials)
        return client.open(sheet_name).sheet1
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        raise

# Log tokens to the Google Sheet
def log_tokens_to_sheet(query, tokens_used,response):
    try:
        # Connect to the sheet
        sheet_name = "GPT_log"
        sheet = connect_to_google_sheet(sheet_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, query, response, tokens_used])
    except Exception as e:
        st.error(f"Failed to log tokens: {e}")
        print(f"Error: {e}")

def log_feedback_to_sheet(feedback, query, response):
    if feedback is None or not query:
        #st.warning("Feedback or query is missing!")
        return
    try:
        # Map feedback values
        if feedback == 0:
            feedback_mapped = "Negative"
        elif feedback == 1:
            feedback_mapped = "Positive"
        else:
            feedback_mapped = "Not Provided"

        # Connect to the sheet
        sheet_name = "Feedback_log"
        sheet = connect_to_google_sheet(sheet_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, query, response, feedback_mapped])

        st.success("Thank you for your feedback!")
    except Exception as e:
        logging.exception("Error while logging feedback")
        st.error(f"Failed to log feedback: {e}")
    
    st.session_state["feedback_submitted"] = False
#-------------------- Drive end -------------------- 


# ------------------- Core Logic Functions -------------------
def search_pinecone(index, embedding):
    """
    Perform a search in Pinecone using the query embedding.
    """
    try:
        # Perform the search
        response = index.query(
            namespace="",             # Search within the default namespace
            vector=embedding,  # Convert embedding to a list format
            top_k=8,                  # Retrieve the top 5 matches
            include_metadata=True     # Include metadata for matched vectors
        )

        # Extract all matches
        all_matches = response

        # Filter matches with a similarity score >= 65%
        #filtered_matches = [match for match in response['matches'] if match['score'] >= 0.65]

        return all_matches
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        print(f"Error querying Pinecone: {e}")
        return {"matches": []}
    
def get_gpt4_response(retrieved_texts, query, max_tokens, temperature):
    """
    Get a response from GPT-4 based on the provided prompt.
    """
     # Refine prompt structure dynamically
    prompt = (
        f"Context: {' '.join(retrieved_texts)}\n\n"
        f"User Query: {query}\n\n"
        "Task:\n"
            f"- {response_type_instruction}\n"
            "- Analyze the provided Context to address the User Query effectively."
            "- Use bullet points for clarity if multiple aspects are present."
            "- Ensure if need then combine all given context to answer the User query."
            "- If the Context provides indirect or scattered details, synthesize them to form a coherent answer."
            "- If the Context does not explicitly mention the answer, provide reasonable inferences based on the available details."
            "- If the Context contains no relevant information, respond with: 'Context does not provide sufficient information to answer the question.'\n\n"
            "- Ensure that sentences with the same meaning are not repeated in the response."
            "Note: Ensure that the response is **strictly based on the provided Context** and avoid introducing external information."
    )

    try:
        gpt4_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant that provides answers based on given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return gpt4_response['choices'][0]['message']['content'], gpt4_response['usage']['total_tokens']
    except Exception as e:
        st.error(f"Error generating response with GPT-4: {e}")
        print(f"Error generating response with GPT-4: {e}")
        return "No response generated.", 0
    
# def log_response(query, tokens_used):
#     """
#     Log the query and response details to a file.
#     """
#     with open("gpt_token_log.txt", "a") as log_file:
#         log_file.write(f"{datetime.now()} | Query: {query} | Tokens Used: {tokens_used}\n")


def display_documents(retrieved_pdf_title, retrieved_pdf_page, retrieved_pdf_link, Context=""):
    """
    Display retrieved documents in a dataframe.
    """
    df = pd.DataFrame({
        "Pdf": retrieved_pdf_title,
        "Page": retrieved_pdf_page,
        "Link": retrieved_pdf_link
    })
    st.dataframe(
        df,
        column_config={
            "Pdf": "Pdf",
            "Page": "Page",
            "Link": st.column_config.LinkColumn("Link"),
        },
        hide_index=True,
    )
    st.write(Context)

# def handle_feedback(feedback, query):
#     """
#     Handle feedback and log it.
#     """
#     print(f"Handling feedback: {feedback}, query: {query}")  # Debug print
#     try:
#         with open("feedback_log.txt", "a") as feedback_log:
#             feedback_log.write(
#                 f"{datetime.now()} | Query: {query} | Feedback: {feedback} \n"
#             )
#         print("Feedback successfully written to file")  # Debug print
#         #st.write("Thank you for your feedback!")
#     except Exception as e:
#         st.error(f"Error writing feedback to file: {e}")
#         print(f"Error writing feedback to file: {e}")  # Debug print
#     st.session_state["feedback_submitted"] = False  # Reset feedback submission status

# ------------------- Main App Logic -------------------

# Load configuration
#config = load_config("config.json")

# Initialize APIs and models
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
index = pc.Index("test-larg-openai")
openai.api_key = st.secrets["openai_key"]

# Streamlit UI setup
st.title(":blue[Chatbot with Pinecone and GPT]")
initialize_session_state()
#clear_text_area()

# Query input and submission buttons
query_input = st.text_area("Enter your query:", value=st.session_state["query_input"], key="query_input")

if 'submit_clicked' not in st.session_state:
    st.session_state.submit_clicked = False

#for ensure when submit button clicked then all app state reset to Empty state
def on_button_click():
    submit_text()
    st.session_state.submit_clicked = True

# Create two columns
col1,col2,col3,col4 = st.columns(4)
row2_col1, row2_col2 = st.columns([3, 1])  # Adjust column width for alignment

# Place buttons in each column
with col1:
    clear_button = st.button("Clear Query", on_click=clear_text_area, key="clear_query")
with col4:
    st.button("Submit Query", on_click=on_button_click)
    
# Add radio buttons for response type in the second row
with row2_col1:
    response_type = st.radio(
        "Select response type:",
        options=["Short", "Long"],
        key="response_type",
        horizontal=True,  # Display options in a row
        index=0
    )

# Adjust token size and response type variable based on selection
if response_type == "Short":
    max_tokens = 800
    response_type_instruction = "Provide a short answer."
else:  # Long response
    max_tokens = 4096  # Max limit for GPT-4 model
    response_type_instruction = "Provide a detailed answer."

if st.session_state.submit_clicked:
        st.session_state.query = query_input
        if st.session_state.query is not None and st.session_state.query !="":
            if st.session_state.feedback is None:
                query = st.session_state.query
                print(f"Query: {query}")  # Debug print
                laser = Laser(
                    "laser_models/93langs.fcodes",
                    "laser_models/93langs.fvocab",
                    "laser_models/bilstm.93langs.2018-12-26.pt"
                )
                
                embedding1 = openai.embeddings.create(
                          input=query,
                          model="text-embedding-3-large"
                      )

                #embedding1=laser.embed_sentences(query, lang='en').tolist()  # Specify the language of the query

                answer = search_pinecone(index, embedding1)

                st.session_state.retrieved_texts = [match['metadata']['text'] for match in answer['matches']]
                st.session_state.retrieved_pdf_title = [match['metadata']['pdf_name'] for match in answer['matches']]
                st.session_state.retrieved_pdf_page = [match['metadata']['page_range'] for match in answer['matches']]
                st.session_state.retrieved_pdf_link = [match['metadata']['link'] for match in answer['matches']]
            
            if st.session_state.retrieved_texts is not None:
                if st.session_state.response is None:
                    response_text, tokens_used = get_gpt4_response(st.session_state["retrieved_texts"], query, max_tokens, temperature=st.secrets["temperature"])
                    st.session_state.response = response_text
                    log_tokens_to_sheet(query, tokens_used,response_text)
                    st.write(st.session_state.response)
                else:
                    response_text=st.session_state["response"]
                    if response_text:
                        st.session_state.response = response_text
                        st.write(st.session_state.response)

                # Feedback code 
                #st.subheader("Feed Back:")
                st.info('Was this answer helpful to you?')
                sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
                st.session_state.feedback = st.feedback("thumbs")
                log_feedback_to_sheet(st.session_state.feedback,st.session_state.query,st.session_state.response)
                
                st.subheader("YOU CAN REFER TO THESE DOCUMENTS:")
                display_documents(st.session_state.retrieved_pdf_title, st.session_state.retrieved_pdf_page, st.session_state.retrieved_pdf_link)
                st.write(st.session_state.retrieved_texts)
    
            else:
                st.write("Pinecone has no relevant context.")
        
        else:
            st.warning('Enter the Query', icon="⚠️")


