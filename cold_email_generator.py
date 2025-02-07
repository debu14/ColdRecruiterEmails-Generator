import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Use API key from .env

# Streamlit UI
st.title("Cold Email Generator for Recruiters")
st.write("Generate personalized cold emails tailored to job descriptions.")

# Input fields
job_description = st.text_area("Paste the Job Description Here:", height=200)
your_name = st.text_input("Your Name:")
your_experience = st.text_input("Your Experience (e.g., '5 years in marketing'):")
your_skills = st.text_input("Your Key Skills (e.g., 'SEO, Content Strategy'):")
company_name = st.text_input("Company Name:")
recruiter_name = st.text_input("Recruiter's Name (if known):")
platform = st.text_input("Where did you find the job? (e.g., LinkedIn, Company Website):")

# Load and process job description
def process_job_description(job_description):
    # Save job description to a temporary file
    with open("job_description.txt", "w") as f:
        f.write(job_description)
    
    # Load and split the document
    loader = TextLoader("job_description.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# Generate cold email using LangChain
def generate_cold_email(db, your_name, your_experience, your_skills, company_name, recruiter_name, platform):
    # Create a retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    # Define the prompt
    prompt = f"""
    Generate a professional cold email for a recruiter based on the following details:
    - Your Name: {your_name}
    - Your Experience: {your_experience}
    - Your Skills: {your_skills}
    - Company Name: {company_name}
    - Recruiter's Name: {recruiter_name}
    - Platform: {platform}
    
    The email should:
    1. Be personalized and concise.
    2. Highlight your skills and experience relevant to the job description.
    3. Include a call to action (e.g., request for a call or meeting).
    4. Be professional and polite.
    """
    
    # Generate the email
    response = qa.run(prompt)
    return response

# Main function
def main():
    if st.button("Generate Cold Email"):
        if not job_description or not your_name or not your_experience or not your_skills or not company_name:
            st.error("Please fill in all required fields.")
        else:
            with st.spinner("Generating your cold email..."):
                # Process job description
                db = process_job_description(job_description)
                
                # Generate cold email
                email = generate_cold_email(db, your_name, your_experience, your_skills, company_name, recruiter_name, platform)
                
                # Display the email
                st.subheader("Your Cold Email:")
                st.write(email)

# Run the app
if __name__ == "__main__":
    main()
