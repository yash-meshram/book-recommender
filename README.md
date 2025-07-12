# Book Recommender System

A sophisticated book recommendation system that uses semantic search, sentiment analysis, and machine learning to provide personalized book recommendations based on user queries, categories, and emotional tones.

## 🚀 Features

### Core Functionality
- **Semantic Book Search**: Find books based on natural language descriptions
- **Category Filtering**: Filter recommendations by book categories (Fiction, Non-Fiction, etc.)
- **Tone-based Sorting**: Sort recommendations by emotional tone (joy, sadness, anger, fear, etc.)
- **Interactive Web Interface**: Beautiful Gradio-based dashboard for easy interaction
- **Vector Database**: Chroma vector store for efficient similarity search

### Advanced Features
- **Sentiment Analysis**: Analyzes emotional content of book descriptions
- **Text Classification**: Categorizes books using zero-shot classification
- **Data Processing Pipeline**: Comprehensive data cleaning and preprocessing
- **Cover Image Integration**: Displays book covers with fallback images

## 📁 Project Structure

```
book-recommender/
├── gradio_dashboard.py          # Main web application
├── data_extraction.ipynb        # Data processing and extraction
├── sentiment_analysis.ipynb     # Emotional tone analysis
├── text_classification.ipynb    # Book categorization
├── vector_search.ipynb          # Vector database setup and search
├── books_final.csv              # Processed book dataset
├── books_cleaned.csv            # Cleaned book data
├── books_with_categories.csv    # Categorized book data
├── tagged_description.txt       # Book descriptions with ISBN tags
├── chroma_books_db/            # Vector database storage
├── cover-not-found.jpg          # Default cover image
└── README.md                    # This file
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.x**: Main programming language
- **Gradio**: Web interface framework
- **LangChain**: LLM integration and document processing
- **Chroma**: Vector database for similarity search
- **HuggingFace**: Embeddings and text models

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities

### Models Used
- **HuggingFace Embeddings**: For text vectorization
- **Zero-shot Classification**: For book categorization
- **Sentiment Analysis**: For emotional tone detection

## 📊 Data Pipeline

### 1. Data Extraction (`data_extraction.ipynb`)
- Extracts book data from various sources
- Cleans and preprocesses book information
- Handles missing data and formatting issues

### 2. Text Classification (`text_classification.ipynb`)
- Uses zero-shot classification to categorize books
- Assigns categories like Fiction, Non-Fiction, etc.
- Creates structured category labels

### 3. Sentiment Analysis (`sentiment_analysis.ipynb`)
- Analyzes emotional content of book descriptions
- Extracts tone scores (joy, sadness, anger, fear, disgust, surprise, neutral)
- Creates emotion-based features for recommendation

### 4. Vector Database (`vector_search.ipynb`)
- Creates embeddings for book descriptions
- Builds Chroma vector database
- Implements similarity search functionality

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install gradio
   pip install langchain
   pip install langchain-community
   pip install langchain-huggingface
   pip install langchain-text-splitters
   pip install chromadb
   pip install pandas
   pip install numpy
   pip install python-dotenv
   pip install transformers
   pip install torch
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   # Add any required API keys or configuration
   ```

## 🎯 Usage

### Running the Web Application

1. **Start the Gradio dashboard**
   ```bash
   python gradio_dashboard.py
   ```

2. **Access the web interface**
   - Open your browser and go to the URL shown in the terminal
   - Typically: `http://localhost:7860`

### Using the Recommendation System

1. **Enter a book description** in the text box
   - Example: "A story about space travel"
   - Example: "A mystery novel with detectives"

2. **Select a category** (optional)
   - Choose from available book categories
   - Use "All" to include all categories

3. **Choose a tone** (optional)
   - Select emotional tone for sorting
   - Options: joy, sadness, anger, fear, disgust, surprise, neutral
   - Use "All" to ignore tone sorting

4. **Click "Recommend Book"** to get personalized recommendations

### Output
The system will display:
- Book covers with titles
- Author information
- Brief descriptions
- Sorted by relevance and selected criteria

## 🔧 Configuration

### Database Configuration
- Vector database is stored in `chroma_books_db/`
- Book data is in CSV format for easy processing
- Embeddings are generated using HuggingFace models

### Model Configuration
- Embedding model: HuggingFace embeddings
- Classification: Zero-shot classification
- Sentiment analysis: Text classification models

## 📈 Performance

### Search Capabilities
- **Semantic Search**: Finds books based on meaning, not just keywords
- **Category Filtering**: Fast filtering by book categories
- **Tone-based Sorting**: Emotional content-aware recommendations
- **Scalable**: Handles large book databases efficiently

### Response Time
- Initial search: ~1-2 seconds
- Filtered results: < 1 second
- Real-time recommendations


---

**Note**: This is an LLM-powered book recommendation system that combines semantic search, sentiment analysis, and machine learning to provide personalized book suggestions based on user preferences and emotional content analysis.
