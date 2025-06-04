"""
VoiceCraft: Shaping Strategy with NLP
Complete Python Implementation for Survey Data Analysis

This module provides a comprehensive NLP pipeline for analyzing customer feedback
from customer surveys and other text sources to generate actionable marketing insights.

Required sample CSV columns:
- customer_id: Unique customer identifier
- feedback_text: Raw customer feedback text
- rating: Customer rating (1-5 scale)
- source: Feedback source (survey, social_media, support, etc.)
- timestamp: When feedback was submitted
- age: Customer age
- customer_segment: Customer tier (premium, standard, budget)
- purchase_amount: Purchase value
- days_since_purchase: Days between purchase and feedback

Features:
- Text preprocessing and cleaning
- Sentiment analysis using BERT and ensemble methods
- Topic modeling with LDA
- Named Entity Recognition
- Customer segmentation
- Interactive visualizations
- Marketing insights generation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
import spacy
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Processing
from datetime import datetime, timedelta
import re
import os
import json
from collections import Counter, defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class DataLoader:
    """Load and validate survey data"""
    
    def __init__(self, file_path='survey_data.csv'):
        self.file_path = file_path
        self.required_columns = [
            'customer_id', 'feedback_text', 'rating', 'source', 
            'timestamp', 'age', 'customer_segment', 'purchase_amount', 
            'days_since_purchase'
        ]
    
    def load_data(self):
        """Load survey data from CSV file"""
        try:
            print(f"Loading data from {self.file_path}...")
            df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            # Validate required columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                print(f"Available columns: {list(df.columns)}")
            
            # Basic data validation
            self._validate_data(df)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print("Data validation completed.")
            return df
            
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            print("Please ensure the survey_data.csv file exists in the current directory.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _validate_data(self, df):
        """Validate data quality"""
        print("\n=== Data Quality Report ===")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print("Missing values per column:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check data types
        print(f"\nTotal records: {len(df)}")
        print(f"Unique customers: {df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'}")
        
        # Check feedback text quality
        if 'feedback_text' in df.columns:
            empty_feedback = df['feedback_text'].isnull().sum()
            short_feedback = (df['feedback_text'].str.len() < 10).sum()
            print(f"Empty feedback: {empty_feedback}")
            print(f"Very short feedback (<10 chars): {short_feedback}")
        
        print("=== End Report ===\n")

class TextPreprocessor:
    """Advanced text preprocessing pipeline"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("spaCy model loaded successfully.")
        except OSError:
            print("spaCy English model not found. Please install it using:")
            print("python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs, emails, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+\.\w+|@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_preprocessing(self, text):
        """Advanced preprocessing using spaCy"""
        if not self.nlp or not text:
            return self._fallback_preprocessing(text)
        
        try:
            doc = self.nlp(text)
            
            # Extract lemmatized tokens, remove stop words and punctuation
            tokens = [
                token.lemma_.lower() for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and not token.is_space
                and len(token.text) > 2
                and token.is_alpha
            ]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"spaCy processing failed: {e}")
            return self._fallback_preprocessing(text)
    
    def _fallback_preprocessing(self, text):
        """Fallback preprocessing without spaCy"""
        if not text:
            return ""
        
        # Tokenize and remove stop words
        tokens = nltk.word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words 
            and token.isalpha() 
            and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def process_dataframe(self, df, text_column='feedback_text'):
        """Process entire dataframe"""
        print("Starting text preprocessing...")
        
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found in dataframe")
            return df
        
        # Basic cleaning
        print("Applying basic text cleaning...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Advanced preprocessing
        print("Applying advanced preprocessing...")
        df['processed_text'] = df['cleaned_text'].apply(self.advanced_preprocessing)
        
        # Remove empty processed texts
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 0]
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} records with empty processed text")
        
        print(f"Text preprocessing completed. Final dataset: {len(df)} records")
        return df

class SentimentAnalyzer:
    """Multi-method sentiment analysis"""
    
    def __init__(self):
        self.bert_analyzer = None
        self.vader_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            print("Loading BERT sentiment model...")
            self.bert_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512
            )
            print("BERT model loaded successfully.")
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            print("Will use fallback methods.")
        
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("VADER sentiment analyzer loaded.")
        except Exception as e:
            print(f"Failed to load VADER: {e}")
    
    def analyze_sentiment_bert(self, text):
        """BERT-based sentiment analysis"""
        if not self.bert_analyzer or not text:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'bert'}
        
        try:
            result = self.bert_analyzer(text)[0]
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], result['label'].lower())
            
            return {
                'sentiment': sentiment,
                'confidence': result['score'],
                'method': 'bert'
            }
        except Exception as e:
            print(f"BERT analysis failed for text: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'bert'}
    
    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis"""
        if not self.vader_analyzer or not text:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'vader'}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(scores['compound']),
                'scores': scores,
                'method': 'vader'
            }
        except Exception as e:
            print(f"VADER analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'vader'}
    
    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        if not text:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'textblob'}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(polarity),
                'polarity': polarity,
                'method': 'textblob'
            }
        except Exception as e:
            print(f"TextBlob analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'textblob'}
    
    def ensemble_sentiment(self, text):
        """Ensemble method combining multiple approaches"""
        bert_result = self.analyze_sentiment_bert(text)
        vader_result = self.analyze_sentiment_vader(text)
        textblob_result = self.analyze_sentiment_textblob(text)
        
        # Collect all sentiment predictions
        sentiments = []
        confidences = []
        
        # Weight BERT higher if available
        if bert_result['confidence'] > 0.5:
            sentiments.extend([bert_result['sentiment']] * 2)
            confidences.append(bert_result['confidence'])
        
        sentiments.append(vader_result['sentiment'])
        sentiments.append(textblob_result['sentiment'])
        confidences.extend([vader_result['confidence'], textblob_result['confidence']])
        
        # Majority vote for final sentiment
        sentiment_counts = Counter(sentiments)
        final_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        return {
            'sentiment': final_sentiment,
            'confidence': avg_confidence,
            'individual_results': {
                'bert': bert_result,
                'vader': vader_result,
                'textblob': textblob_result
            }
        }
    
    def analyze_dataframe(self, df, text_column='processed_text'):
        """Analyze sentiment for entire dataframe"""
        print("Starting sentiment analysis...")
        
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found")
            return df
        
        results = []
        total_texts = len(df)
        
        for i, text in enumerate(df[text_column]):
            if i % 1000 == 0:
                print(f"Processed {i}/{total_texts} texts ({i/total_texts*100:.1f}%)")
            
            result = self.ensemble_sentiment(text)
            results.append(result)
        
        # Extract results into separate columns
        df['predicted_sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_confidence'] = [r['confidence'] for r in results]
        df['bert_sentiment'] = [r['individual_results']['bert']['sentiment'] for r in results]
        df['vader_sentiment'] = [r['individual_results']['vader']['sentiment'] for r in results]
        df['textblob_sentiment'] = [r['individual_results']['textblob']['sentiment'] for r in results]
        
        print("Sentiment analysis completed.")
        
        # Print sentiment distribution
        sentiment_dist = df['predicted_sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        return df

class TopicModeler:
    """Advanced topic modeling using LDA"""
    
    def __init__(self, n_topics=15, random_state=42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        self.topic_labels = {}
        
    def fit_transform(self, texts):
        """Fit LDA model and transform texts"""
        print(f"Training topic model with {self.n_topics} topics...")
        
        # Filter out empty texts
        texts = [text for text in texts if text and len(text.strip()) > 0]
        print(f"Processing {len(texts)} valid texts...")
        
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        try:
            dtm = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=10,
                learning_method='online',
                learning_offset=50.0
            )
            
            doc_topic_matrix = self.lda_model.fit_transform(dtm)
            
            # Generate topic labels
            self._generate_topic_labels()
            
            print("Topic modeling completed successfully.")
            return doc_topic_matrix
            
        except Exception as e:
            print(f"Topic modeling failed: {e}")
            return None
    
    def _generate_topic_labels(self):
        """Generate human-readable topic labels"""
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [self.feature_names[i] for i in topic.argsort()[-3:][::-1]]
            self.topic_labels[topic_idx] = ' + '.join(top_words)
    
    def get_topic_words(self, n_words=10):
        """Get top words for each topic"""
        if not self.lda_model:
            return []
        
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [self.feature_names[i] for i in topic.argsort()[-n_words:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'label': self.topic_labels.get(topic_idx, f'Topic {topic_idx}'),
                'words': top_words,
                'weights': topic[topic.argsort()[-n_words:][::-1]]
            })
        return topics
    
    def predict_topics(self, texts):
        """Predict topics for new texts"""
        if not self.vectorizer or not self.lda_model:
            print("Model not trained yet!")
            return None
        
        dtm = self.vectorizer.transform(texts)
        doc_topic_matrix = self.lda_model.transform(dtm)
        return doc_topic_matrix
    
    def get_dominant_topics(self, doc_topic_matrix):
        """Get dominant topic for each document"""
        return np.argmax(doc_topic_matrix, axis=1)
    
    def analyze_dataframe(self, df, text_column='processed_text'):
        """Add topic analysis to dataframe"""
        print("Starting topic analysis...")
        
        if text_column not in df.columns:
            print(f"Error: Column '{text_column}' not found")
            return df
        
        # Fit topic model
        doc_topic_matrix = self.fit_transform(df[text_column].tolist())
        
        if doc_topic_matrix is not None:
            # Get dominant topics
            dominant_topics = self.get_dominant_topics(doc_topic_matrix)
            
            # Add topic information to dataframe
            df['dominant_topic'] = dominant_topics
            df['topic_label'] = [self.topic_labels.get(topic, f'Topic {topic}') 
                               for topic in dominant_topics]
            
            # Add topic probabilities for top 3 topics
            for i in range(min(3, self.n_topics)):
                df[f'topic_{i}_prob'] = doc_topic_matrix[:, i]
            
            print("Topic analysis completed.")
            
            # Print topic distribution
            topic_dist = df['dominant_topic'].value_counts().sort_index()
            print("\nTopic Distribution:")
            for topic_id, count in topic_dist.items():
                label = self.topic_labels.get(topic_id, f'Topic {topic_id}')
                print(f"  Topic {topic_id} ({label}): {count} ({count/len(df)*100:.1f}%)")
        
        return df

class CustomerSegmentation:
    """Customer segmentation based on sentiment and behavior"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.segment_names = {}
        self.feature_names = []
    
    def create_features(self, df):
        """Create features for segmentation"""
        features = pd.DataFrame()
        
        # Sentiment features
        sentiment_dummies = pd.get_dummies(df['predicted_sentiment'], prefix='sentiment')
        features = pd.concat([features, sentiment_dummies], axis=1)
        
        # Behavioral features
        if 'rating' in df.columns:
            features['avg_rating'] = df['rating']
        if 'sentiment_confidence' in df.columns:
            features['sentiment_confidence'] = df['sentiment_confidence']
        if 'purchase_amount' in df.columns:
            features['purchase_amount'] = df['purchase_amount']
        if 'days_since_purchase' in df.columns:
            features['days_since_purchase'] = df['days_since_purchase']
        if 'age' in df.columns:
            features['age'] = df['age']
        
        # Source features
        if 'source' in df.columns:
            source_dummies = pd.get_dummies(df['source'], prefix='source')
            features = pd.concat([features, source_dummies], axis=1)
        
        # Topic features (if available)
        topic_cols = [col for col in df.columns if col.startswith('topic_') and col.endswith('_prob')]
        if topic_cols:
            features = pd.concat([features, df[topic_cols]], axis=1)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def fit_predict(self, df):
        """Perform customer segmentation"""
        print("Performing customer segmentation...")
        
        features = self.create_features(df)
        
        if features.empty:
            print("No features available for segmentation")
            return df
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Add cluster information to dataframe
        df['customer_cluster'] = clusters
        
        # Generate cluster names based on characteristics
        self._generate_cluster_names(df, features)
        
        df['cluster_name'] = [self.segment_names.get(cluster, f'Segment {cluster}') 
                             for cluster in clusters]
        
        # Calculate silhouette score
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(features_scaled, clusters)
            print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        print("Customer segmentation completed.")
        
        # Print cluster distribution
        cluster_dist = df['customer_cluster'].value_counts().sort_index()
        print("\nCustomer Segment Distribution:")
        for cluster_id, count in cluster_dist.items():
            name = self.segment_names.get(cluster_id, f'Segment {cluster_id}')
            print(f"  {name}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _generate_cluster_names(self, df, features):
        """Generate meaningful cluster names"""
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['customer_cluster'] == cluster_id]
            
            # Analyze cluster characteristics
            if len(cluster_data) == 0:
                continue
            
            # Sentiment characteristics
            sentiment_mode = cluster_data['predicted_sentiment'].mode()[0] if not cluster_data['predicted_sentiment'].mode().empty else 'mixed'
            
            # Rating characteristics
            avg_rating = cluster_data['rating'].mean() if 'rating' in cluster_data.columns else 0
            
            # Purchase behavior
            avg_purchase = cluster_data['purchase_amount'].mean() if 'purchase_amount' in cluster_data.columns else 0
            
            # Generate name based on characteristics
            if sentiment_mode == 'positive' and avg_rating >= 4:
                if avg_purchase > df['purchase_amount'].median():
                    name = "Premium Advocates"
                else:
                    name = "Satisfied Customers"
            elif sentiment_mode == 'negative' or avg_rating <= 2:
                name = "At-Risk Customers"
            elif avg_purchase > df['purchase_amount'].quantile(0.75):
                name = "High-Value Customers"
            elif avg_purchase < df['purchase_amount'].quantile(0.25):
                name = "Budget Customers"
            else:
                name = f"Standard Segment {cluster_id}"
            
            self.segment_names[cluster_id] = name

class NLPInsights:
    """Generate business insights from NLP analysis"""
    
    def __init__(self):
        self.insights = {}
    
    def generate_sentiment_insights(self, df):
        """Generate sentiment-based insights"""
        insights = {}
        
        # Overall sentiment distribution
        sentiment_dist = df['predicted_sentiment'].value_counts(normalize=True)
        insights['sentiment_distribution'] = sentiment_dist.to_dict()
        
        # Sentiment by source
        if 'source' in df.columns:
            sentiment_by_source = df.groupby('source')['predicted_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
            insights['sentiment_by_source'] = sentiment_by_source.to_dict()
        
        # Sentiment trends over time
        if 'timestamp' in df.columns:
            df['month'] = df['timestamp'].dt.to_period('M')
            sentiment_trends = df.groupby('month')['predicted_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
            insights['sentiment_trends'] = sentiment_trends.to_dict()
        
        # Rating vs sentiment correlation
        if 'rating' in df.columns:
            rating_sentiment = df.groupby('rating')['predicted_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
            insights['rating_sentiment_correlation'] = rating_sentiment.to_dict()
        
        return insights
    
    def generate_topic_insights(self, df, topic_modeler):
        """Generate topic-based insights"""
        insights = {}
        
        if 'dominant_topic' not in df.columns:
            return insights
        
        # Topic frequency
        topic_freq = df['dominant_topic'].value_counts()
        insights['topic_frequency'] = {
            topic_modeler.topic_labels.get(topic_id, f'Topic {topic_id}'): count 
            for topic_id, count in topic_freq.items()
        }
        
        # Sentiment by topic
        topic_sentiment = df.groupby('dominant_topic')['predicted_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
        insights['sentiment_by_topic'] = {
            topic_modeler.topic_labels.get(topic_id, f'Topic {topic_id}'): sentiment_dist.to_dict()
            for topic_id, sentiment_dist in topic_sentiment.iterrows()
        }
        
        # Topic words
        topic_words = topic_modeler.get_topic_words(n_words=5)
        insights['topic_words'] = {
            f"Topic {topic['topic_id']} ({topic['label']})": topic['words']
            for topic in topic_words
        }
        
        return insights
    
    def generate_customer_insights(self, df):
        """Generate customer segmentation insights"""
        insights = {}
        
        if 'customer_cluster' not in df.columns:
            return insights
        
        # Segment characteristics
        segment_stats = df.groupby('cluster_name').agg({
            'predicted_sentiment': lambda x: x.value_counts().index[0],  # Most common sentiment
            'rating': 'mean',
            'purchase_amount': 'mean',
            'sentiment_confidence': 'mean',
            'age': 'mean'
        }).round(2)
        
        insights['segment_characteristics'] = segment_stats.to_dict()
        
        # Segment sizes
        segment_sizes = df['cluster_name'].value_counts()
        insights['segment_sizes'] = segment_sizes.to_dict()
        
        return insights
    
    def generate_business_recommendations(self, df, topic_modeler):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Sentiment-based recommendations
        negative_pct = (df['predicted_sentiment'] == 'negative').mean()
        if negative_pct > 0.3:
            recommendations.append({
                'priority': 'High',
                'category': 'Customer Satisfaction',
                'recommendation': f'Address customer satisfaction issues - {negative_pct:.1%} negative sentiment detected',
                'action': 'Implement immediate customer service improvements and follow-up programs'
            })
        
        # Topic-based recommendations
        if 'dominant_topic' in df.columns:
            # Find most problematic topics (high negative sentiment)
            topic_sentiment = df.groupby('dominant_topic')['predicted_sentiment'].apply(
                lambda x: (x == 'negative').mean()
            ).sort_values(ascending=False)
            
            for topic_id, neg_rate in topic_sentiment.head(3).items():
                if neg_rate > 0.4:
                    topic_label = topic_modeler.topic_labels.get(topic_id, f'Topic {topic_id}')
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Product/Service Improvement',
                        'recommendation': f'Address issues in "{topic_label}" - {neg_rate:.1%} negative sentiment',
                        'action': f'Investigate and improve aspects related to {topic_label.lower()}'
                    })
        
        # Segmentation-based recommendations
        if 'customer_cluster' in df.columns:
            at_risk_segments = df[df['predicted_sentiment'] == 'negative']['cluster_name'].value_counts()
            if not at_risk_segments.empty:
                top_at_risk = at_risk_segments.index[0]
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Customer Retention',
                    'recommendation': f'Focus retention efforts on "{top_at_risk}" segment',
                    'action': 'Develop targeted retention campaigns and personalized outreach'
                })
        
        return recommendations
    
    def generate_full_report(self, df, topic_modeler):
        """Generate comprehensive business insights report"""
        print("Generating comprehensive insights report...")
        
        insights = {}
        
        # Sentiment insights
        insights['sentiment'] = self.generate_sentiment_insights(df)
        
        # Topic insights
        insights['topics'] = self.generate_topic_insights(df, topic_modeler)
        
        # Customer insights
        insights['customers'] = self.generate_customer_insights(df)
        
        # Business recommendations
        insights['recommendations'] = self.generate_business_recommendations(df, topic_modeler)
        
        # Topic modeling output
        insights['topic_output'] = self.generate_topic_output(df, topic_modeler)
        
        self.insights = insights
        return insights
    
    def generate_topic_output(self, df, topic_modeler):
        """Generate detailed topic modeling output for business stakeholders"""
        output = {}
        
        if 'dominant_topic' not in df.columns or not topic_modeler.lda_model:
            return output
        
        # Get topic words and labels
        topic_words = topic_modeler.get_topic_words(n_words=10)
        
        # Calculate topic statistics
        topic_stats = []
        for topic_info in topic_words:
            topic_id = topic_info['topic_id']
            topic_label = topic_info['label']
            
            # Filter data for this topic
            topic_data = df[df['dominant_topic'] == topic_id]
            
            if len(topic_data) == 0:
                continue
            
            # Calculate sentiment distribution
            sentiment_dist = topic_data['predicted_sentiment'].value_counts(normalize=True)
            avg_sentiment_score = topic_data['sentiment_confidence'].mean()
            
            # Calculate average rating if available
            avg_rating = topic_data['rating'].mean() if 'rating' in topic_data.columns else None
            
            # Identify if this is a problem area (high negative sentiment)
            negative_pct = sentiment_dist.get('negative', 0) * 100
            
            # Categorize issue severity
            if negative_pct > 60:
                severity = "Critical"
                priority = "High"
            elif negative_pct > 40:
                severity = "Moderate"
                priority = "Medium"
            elif negative_pct > 20:
                severity = "Minor"
                priority = "Low"
            else:
                severity = "Positive"
                priority = "Maintain"
            
            topic_stats.append({
                'topic_id': topic_id,
                'topic_label': topic_label,
                'keywords': topic_info['words'][:5],
                'frequency_pct': len(topic_data) / len(df) * 100,
                'total_mentions': len(topic_data),
                'avg_sentiment_score': avg_sentiment_score,
                'negative_pct': negative_pct,
                'positive_pct': sentiment_dist.get('positive', 0) * 100,
                'neutral_pct': sentiment_dist.get('neutral', 0) * 100,
                'avg_rating': avg_rating,
                'severity': severity,
                'priority': priority,
                'sample_feedback': topic_data['feedback_text'].iloc[:3].tolist() if 'feedback_text' in topic_data.columns else []
            })
        
        # Sort by frequency (most common topics first)
        topic_stats.sort(key=lambda x: x['frequency_pct'], reverse=True)
        
        output['topic_analysis'] = topic_stats
        
        # Generate issue categorization
        critical_issues = [t for t in topic_stats if t['severity'] == 'Critical']
        moderate_issues = [t for t in topic_stats if t['severity'] == 'Moderate']
        positive_topics = [t for t in topic_stats if t['severity'] == 'Positive']
        
        output['issue_summary'] = {
            'critical_issues': len(critical_issues),
            'moderate_issues': len(moderate_issues),
            'positive_topics': len(positive_topics),
            'total_topics': len(topic_stats)
        }
        
        # Top issues for immediate attention
        output['top_issues'] = sorted(
            [t for t in topic_stats if t['severity'] in ['Critical', 'Moderate']], 
            key=lambda x: x['negative_pct'], 
            reverse=True
        )[:5]
        
        # Top positive aspects to leverage
        output['top_positives'] = sorted(
            [t for t in topic_stats if t['severity'] == 'Positive'], 
            key=lambda x: x['positive_pct'], 
            reverse=True
        )[:3]
        
        return output
    
    def print_topic_report(self, insights):
        """Print a formatted topic analysis report"""
        if 'topic_output' not in insights:
            print("No topic output available")
            return
        
        topic_output = insights['topic_output']
        
        print("\n" + "="*80)
        print("📊 TOPIC MODELING ANALYSIS REPORT")
        print("="*80)
        
        # Summary statistics
        if 'issue_summary' in topic_output:
            summary = topic_output['issue_summary']
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Total Topics Identified: {summary['total_topics']}")
            print(f"   Critical Issues: {summary['critical_issues']}")
            print(f"   Moderate Issues: {summary['moderate_issues']}")
            print(f"   Positive Topics: {summary['positive_topics']}")
        
        # Top issues requiring attention
        print(f"\n🚨 TOP ISSUES REQUIRING IMMEDIATE ATTENTION:")
        print("-" * 60)
        if 'top_issues' in topic_output:
            for i, issue in enumerate(topic_output['top_issues'], 1):
                print(f"{i}. {issue['topic_label'].upper()}")
                print(f"   📊 Frequency: {issue['frequency_pct']:.1f}% ({issue['total_mentions']:,} mentions)")
                print(f"   😠 Negative Sentiment: {issue['negative_pct']:.1f}%")
                print(f"   🏷️  Keywords: {', '.join(issue['keywords'])}")
                print(f"   ⚠️  Severity: {issue['severity']} | Priority: {issue['priority']}")
                if issue['sample_feedback']:
                    print(f"   💬 Sample: \"{issue['sample_feedback'][0][:100]}...\"")
                print()
        
        # Top positive aspects
        print(f"\n✅ TOP POSITIVE ASPECTS TO LEVERAGE:")
        print("-" * 60)
        if 'top_positives' in topic_output:
            for i, positive in enumerate(topic_output['top_positives'], 1):
                print(f"{i}. {positive['topic_label'].upper()}")
                print(f"   📊 Frequency: {positive['frequency_pct']:.1f}% ({positive['total_mentions']:,} mentions)")
                print(f"   😊 Positive Sentiment: {positive['positive_pct']:.1f}%")
                print(f"   🏷️  Keywords: {', '.join(positive['keywords'])}")
                if positive['sample_feedback']:
                    print(f"   💬 Sample: \"{positive['sample_feedback'][0][:100]}...\"")
                print()
        
        # All topics overview
        print(f"\n📋 ALL TOPICS OVERVIEW:")
        print("-" * 80)
        print(f"{'Topic':<25} {'Freq %':<8} {'Neg %':<8} {'Pos %':<8} {'Severity':<10} {'Priority':<8}")
        print("-" * 80)
        
        if 'topic_analysis' in topic_output:
            for topic in topic_output['topic_analysis']:
                topic_name = topic['topic_label'][:24]
                print(f"{topic_name:<25} {topic['frequency_pct']:<7.1f}% {topic['negative_pct']:<7.1f}% "
                      f"{topic['positive_pct']:<7.1f}% {topic['severity']:<10} {topic['priority']:<8}")
        
        print("="*80)

class VoiceCraftAnalyzer:
    """Main class to orchestrate the entire NLP analysis pipeline"""
    
    def __init__(self, csv_file_path='survey_data.csv'):
        self.csv_file_path = csv_file_path
        self.data_loader = DataLoader(csv_file_path)
        self.text_preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler(n_topics=15)
        self.customer_segmentation = CustomerSegmentation(n_clusters=5)
        self.insights_generator = NLPInsights()
        
        self.df = None
        self.insights = None
    
    def run_full_analysis(self):
        """Run the complete NLP analysis pipeline"""
        print("🎯 Starting VoiceCraft NLP Analysis Pipeline...")
        print("="*60)
        
        # Step 1: Load data
        self.df = self.data_loader.load_data()
        if self.df is None:
            print("❌ Failed to load data. Exiting...")
            return None
        
        # Step 2: Text preprocessing
        self.df = self.text_preprocessor.process_dataframe(self.df)
        
        # Step 3: Sentiment analysis
        self.df = self.sentiment_analyzer.analyze_dataframe(self.df)
        
        # Step 4: Topic modeling
        self.df = self.topic_modeler.analyze_dataframe(self.df)
        
        # Step 5: Customer segmentation
        self.df = self.customer_segmentation.fit_predict(self.df)
        
        # Step 6: Generate insights
        self.insights = self.insights_generator.generate_full_report(self.df, self.topic_modeler)
        
        print("\n✅ Analysis completed successfully!")
        print("="*60)
        
        return self.df, self.insights
    
    def export_results(self, output_prefix='voicecraft_results'):
        """Export analysis results to files"""
        if self.df is None:
            print("No analysis results to export")
            return
        
        # Export processed data
        output_file = f"{output_prefix}_processed_data.csv"
        self.df.to_csv(output_file, index=False)
        print(f"📁 Processed data exported to: {output_file}")
        
        # Export insights as JSON
        if self.insights:
            insights_file = f"{output_prefix}_insights.json"
            with open(insights_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                import json
                insights_json = json.dumps(self.insights, default=str, indent=2)
                f.write(insights_json)
            print(f"📁 Insights exported to: {insights_file}")
        
        # Export topic report
        self.generate_topic_report(output_prefix)
    
    def generate_topic_report(self, output_prefix='voicecraft_results'):
        """Generate and save detailed topic analysis report"""
        if not self.insights:
            print("No insights available for report generation")
            return
        
        report_file = f"{output_prefix}_topic_report.txt"
        
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        
        with open(report_file, 'w') as f:
            sys.stdout = f
            self.insights_generator.print_topic_report(self.insights)
        
        sys.stdout = original_stdout
        print(f"📁 Topic report exported to: {report_file}")
    
    def visualize_results(self):
        """Create visualizations for the analysis results"""
        if self.df is None:
            print("No data available for visualization")
            return
        
        print("📊 Generating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VoiceCraft NLP Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution
        sentiment_counts = self.df['predicted_sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Top topics
        if 'dominant_topic' in self.df.columns:
            topic_counts = self.df['dominant_topic'].value_counts().head(10)
            topic_labels = [self.topic_modeler.topic_labels.get(topic_id, f'Topic {topic_id}') 
                           for topic_id in topic_counts.index]
            axes[0, 1].barh(range(len(topic_counts)), topic_counts.values)
            axes[0, 1].set_yticks(range(len(topic_counts)))
            axes[0, 1].set_yticklabels([label[:20] + '...' if len(label) > 20 else label 
                                       for label in topic_labels])
            axes[0, 1].set_title('Top 10 Topics by Frequency')
            axes[0, 1].set_xlabel('Number of Mentions')
        
        # 3. Sentiment by source
        if 'source' in self.df.columns:
            sentiment_by_source = pd.crosstab(self.df['source'], self.df['predicted_sentiment'], normalize='index')
            sentiment_by_source.plot(kind='bar', ax=axes[1, 0], stacked=True)
            axes[1, 0].set_title('Sentiment Distribution by Source')
            axes[1, 0].set_xlabel('Feedback Source')
            axes[1, 0].set_ylabel('Proportion')
            axes[1, 0].legend(title='Sentiment')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Customer segments
        if 'customer_cluster' in self.df.columns:
            cluster_counts = self.df['customer_cluster'].value_counts().sort_index()
            cluster_names = [self.customer_segmentation.segment_names.get(cluster_id, f'Segment {cluster_id}') 
                            for cluster_id in cluster_counts.index]
            axes[1, 1].bar(range(len(cluster_counts)), cluster_counts.values)
            axes[1, 1].set_xticks(range(len(cluster_counts)))
            axes[1, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                       for name in cluster_names], rotation=45)
            axes[1, 1].set_title('Customer Segment Distribution')
            axes[1, 1].set_ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.savefig('voicecraft_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📁 Visualization saved as: voicecraft_analysis_results.png")

def main():
    """Main function to run the VoiceCraft analysis"""
    print("🎯 Welcome to VoiceCraft: NLP-Driven Marketing Optimization")
    print("="*70)
    
    # Initialize analyzer
    analyzer = VoiceCraftAnalyzer('survey_data.csv')
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    if results:
        df, insights = results
        
        # Print topic report to console
        analyzer.insights_generator.print_topic_report(insights)
        
        # Export results
        analyzer.export_results()
        
        # Generate visualizations
        analyzer.visualize_results()
        
        print("\n🎉 VoiceCraft analysis completed successfully!")
        print("📁 Check the generated files for detailed results and insights.")
    
    else:
        print("❌ Analysis failed. Please check your data file and try again.")

if __name__ == "__main__":
    main()