import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
import streamlit as st
import re
from typing import List, Dict, Tuple
import pickle
from pathlib import Path

# Load spaCy model
nlp = spacy.load('en_core_web_sm')