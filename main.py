import chromadb
import yaml

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import boto3

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import Normalize

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from PIL import Image

from CXR_ReDonE_dataset import CXRTestDataset

from models.vit import interpolate_pos_embed
from models.model_retrieval import ALBEF as ALBEF_Retrieval
from transformers import BertTokenizer

import tempfile
import os


import streamlit as st


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = yaml.load(open('configs/Retrieval_flickr.yaml', 'r'), Loader=yaml.Loader)

    model = ALBEF_Retrieval(
        config=config,
        text_encoder='bert-base-uncased',
        tokenizer=tokenizer
    ).to(device=device)
    checkpoint = torch.load('ALBEF_4M.pth', map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    m_pos_embed_reshaped = interpolate_pos_embed(
        state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m
    )
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model = model.eval()
    return model


def preprocess_image(model, image_path):
    image_embedding_list = []

    transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=Image.BICUBIC),
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
    ])
    dset = CXRTestDataset(transform=transform, target_files=image_path)
    for i in tqdm(range(len(dset))):
        images = dset[i].to('cpu', dtype=torch.float)
        images = torch.unsqueeze(images, axis=0)
        image_features = model.visual_encoder(images)
        image_features = model.vision_proj(image_features[:, 0, :])
        image_features = F.normalize(image_features, dim=-1)

        # Flatten the embedding to a list, as required by Chroma
        image_embedding_list = image_features.tolist()

    return image_embedding_list


def download_vector_db():
    s3_client = boto3.resource(
        's3',
        aws_access_key_id=st.secrets.aws_access_key_id,
        aws_secret_access_key=st.secrets.aws_secret_access_key
    )
    s3_folder = "testRagd"
    local_dir = "vectorstores"
    bucket = s3_client.Bucket(st.secrets.aws_bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        # Check if file already exists
        if not os.path.exists(target):
            print(f"Downloading {obj.key}")
            bucket.download_file(obj.key, target)
        else:
            print(f"File {target} already exists, skipping download.")

    # s3_object = s3_client.get_object(Bucket=st.secrets.aws_bucket_name, Key="testRagd/chroma.sqlite3")
    # body = s3_object["Body"]
    #
    # # Ensure the local directory exists
    # local_directory = "vectorstores"
    # if not os.path.exists(local_directory):
    #     os.makedirs(local_directory)
    #
    # # Define the local file path
    # local_file_path = os.path.join(local_directory, "testRagd")
    #
    # # Write the data to a local file
    # with open(local_file_path, 'wb') as file:
    #     file.write(body.read())
    #
    # # Confirm download
    # print(f"File downloaded to {local_file_path}")


def retreive_vector_db(image_embedding_list):
    chroma_client = chromadb.PersistentClient(path="vectorstores")
    collection = chroma_client.get_collection(name="my_collection")

    # Query the collection with the correct embedding
    query_results = collection.query(
        query_embeddings=image_embedding_list,
        n_results=10,
        include=['documents', 'distances', 'metadatas', 'data', 'uris'],
    )
    return query_results


def generate_reports(context):
    template = """You are an assistant designed to write impression summaries for the radiology report. 
            Users will send a context text and you will respond with an impression summary using that context.
            Instructions:
            • Impression should be based on the information that the user will send in the context.
            • The impression should not mention anything about follow-up actions.
            • Impression should not contain any mentions of prior or previous studies.
            • Limit the generation to maxlen words.
            • Use bullet points and never repeat findings
            • Do it in Indonesian Language (do not translate medical term or latin)
            Context: {context} 

            Impression summary:
            """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0,
                     openai_api_key=st.secrets["openai_api_key"])

    rag_chain = (
            {"context": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    result = rag_chain.invoke(context)
    print(result)
    return result


def main():
    download_vector_db()

    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img, width=500)

        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

        model = load_model()

        image_path = [path]
        image_embedding_list = preprocess_image(model, image_path)

        query_results = retreive_vector_db(image_embedding_list)

        report = generate_reports(query_results['documents'])
        st.write(report)


if __name__ == '__main__':
    main()
