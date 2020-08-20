import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


def clean_transform_title(job_title):
    """Clean and transform job title. Remove punctuations, special characters,
    multiple spaces etc.
    """
    if not isinstance(job_title, str):
        return ''
    new_job_title = job_title.lower()
    special_characters = re.compile('[^ a-zA-Z]')
    new_job_title = re.sub(special_characters, ' ', new_job_title)
    extra_spaces = re.compile(r'\s+')
    new_job_title = re.sub(extra_spaces, ' ', new_job_title)
    
    return new_job_title


class SeniorityModel:
    """Job seniority model class. Contains attributes to fit, predict,
    save and load the job seniority model.
    """
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def _check_for_array(self, variable):
        if not isinstance(variable, (list, tuple, np.ndarray)):
            raise TypeError("variable should be of type list or numpy array.")
        return
    
    def _data_check(self, job_titles, job_seniorities):
        self._check_for_array(job_titles)
        self._check_for_array(job_seniorities)
        
        if len(job_titles) != len(job_seniorities):
            raise IndexError("job_titles and job_seniorities must be of the same length.")
        
        return
        
    def fit(self, job_titles, job_seniorities):
        """Fits the model to predict job seniority from job titles.
        Note that job_titles and job_seniorities must be of the same length.
        
        Parameters
        ----------
        job_titles: numpy array or list of strings representing job titles
        job_seniorities: numpy array or list of strings representing job seniorities
        """
        self._data_check(job_titles, job_seniorities)
        
        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        
        self.vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
        vectorized_data = self.vectorizer.fit_transform(cleaned_job_titles)
        self.model = LinearSVC()
        self.model.fit(vectorized_data, job_seniorities)
        
        return
