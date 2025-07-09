import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_final.csv')

# fixing teh size of the thumbnail
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# if no thumbnail then use cover_not_found image
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Loading books db
persist_directory = "chroma_books_db"

db_books = Chroma(
    persist_directory=persist_directory,
    embedding_function=HuggingFaceEmbeddings()
)

# performing similarity search
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    # taking teh recommendation
    recommendations = db_books.similarity_search(query, k = initial_top_k)

    # extracting isbn13 from descriptions
    isbn13_list = [int(rec.page_content.strip('"').split()[0]) for rec in recommendations]

    # matching extracted isbn13 to books df
    books_recommended = books[books["isbn13"].isin(isbn13_list)].head(initial_top_k)

    # Category wise selection
    if category != "All":
        books_recommended = books_recommended[books_recommended["simple_categories"] == category]

    # Selecting teh top final_kth books (ex.: 16)
    books_recommended = books_recommended.head(final_top_k)

    # Tone wise sorting
    if tone != "All":
        books_recommended.sort_values(by=tone, ascending=False, inplace=True)

    return books_recommended


def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    book_recommendations = retrieve_semantic_recommendations(query, category, tone)

    results = []

    for _, row in book_recommendations.iterrows():
        # description of the book
        description = row["description"]
        description_split = description.split()
        description = " ".join(description_split[:30]) + "..."

        # authors of the book
        authors_split = row["authors"].split(";")
        if len(authors_split) > 1:
            authors = f"{','.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors = f"{authors_split[0]}"

        # Title of the book
        full_title = row['title_subtitle']

        # caption of the book
        caption = f"{full_title} by {authors}: {description}"

        # full result = thumbnail + caption
        results.append((row["large_thumbnail"], caption))

    return results

# Categories dropdown
categories = ["All"] + sorted(books["simple_categories"].unique())

# Tone dropdown
tones = ["All"] + sorted(["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"])

# Building the front-end
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:

    # Title
    gr.Markdown("# Book Recommended")

    with gr.Row():
        # Inputs
        user_query = gr.Textbox(
            label = "About Book",
            placeholder = "eg.: A story about space travel"
        )
        category_dropdown = gr.Dropdown(
            choices = categories,
            label = "Category",
            value = "All"
        )
        tone_dropdown = gr.Dropdown(
            choices = tones,
            label = "Tone",
            value = "All"
        )

        # Submit
        submit_btn = gr.Button("Recommend Book")

    # output
    gr.Markdown("## Book Recommendations:")
    output = gr.Gallery(
        label = "Recommended Books",
        columns = 8, rows = 3
    )

    # When submit button click
    submit_btn.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

if __name__ == "__main__":
    dashboard.launch()

