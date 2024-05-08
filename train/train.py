import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def ingest_data(file_path):
    return pd.read_excel(file_path)

def clean_data(df : pd.DataFrame):
    df = df[['survived', 'pclass', 'sex', 'age']]

    #suppression des lignes avec valeur manquante
    #remplacer les valeur non numérique 
    df.dropna(inplace=True)
    df['sex'] = df['sex'].map({'female': 1, 'male': 0})
    

    
    return df

def train_model(df):

    model = KNeighborsClassifier(n_neighbors=3)
    y = df["survived"]  
    X = df[['pclass', 'sex', 'age']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    score=model.score(X_test,y_test)
    print(f"model score:{score}")
    

    return model

if __name__ == "__main__":
        df =ingest_data("api-model/train/titanic.xls")
        df = clean_data(df)
        model=train_model(df)
        joblib.dump(model,"model_titanic.joblib")