import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataUnderstanding:
    def __init__(self, df):
        self.df = df
        
    def check_head_tail(self):
        print("\n First 5 Rows:")
        print(self.df.head())

        print("\n Last 5 Rows:")
        print(self.df.tail())
    
    def check_shape(self):
        print(f"\n Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def check_columns(self):
        print("\n Column Names:")
        print(list(self.df.columns))
    
    def check_dtypes(self):
        print("\n Data Types:")
        print(self.df.dtypes)
    
    def check_info(self):
        print("\n Dataset Info:")
        print(self.df.info())
    
    def check_summary(self):
        print("\n Summary Statistics:")
        print(self.df.describe())
    
    def full_report(self):
        self.check_head_tail()
        self.check_shape()
        self.check_columns()
        self.check_dtypes()
        self.check_info()
        self.check_summary()
        
        
class EDA:
    def __init__(self, df):
        self.df = df
        
    def histplot(self, columns):
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
            
    def boxplot(self, columns):
        for col in columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()
            
    def heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
        plt.title("Feature Correlation Heatmap")
        plt.show()
        
    def scatterplot(self, columns, column1, column2):
        for col in columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[col], y=self.df[column1])
            plt.title(f"{column1} vs {col}")
            plt.xlabel(col)
            plt.ylabel(f"{column1}")
            plt.show()
            
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[col], y=self.df[column2])
            plt.title(f"{column2} vs {col}")
            plt.xlabel(col)
            plt.ylabel(f"{column2}")
            plt.show()
            
    def pairplot(self):
        sns.pairplot(self.df, corner=True, height=2, aspect=1)
        plt.title(f"Multivariate analysis using Pairplots")
        plt.show()
        
    def countplot(self,column):
        plt.figure(figsize=(12, 5))
        sns.countplot(y=self.df[column], order=self.df[column].value_counts().index[:10])
        plt.title(f"Top 10 Most Frequent {column}")
        plt.show()