#%%
import pandas as pd
from scipy.io import arff
#%%
def load_raw_data(config):
    if config["dataset"] == "abalone":
        data = pd.read_csv('../data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"
    
    elif config["dataset"] == "adult":
        col_names=[
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        base = pd.read_csv('../data/adult.data', 
                           header=None,names = col_names)
        base = base[(base == '?').sum(axis=1) == 0]
        data = base.dropna()
        
        continuous_features = [
            'age', # target variable
            'educational-num',
            'capital-gain', 
            'capital-loss', 
            'hours-per-week',
        ]
        categorical_features = [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native-country',
            'income', # target variable
        ]
        integer_features = continuous_features
        
        ClfTarget = 'income'

    elif config["dataset"] == "abalone":
        data = pd.read_csv('../data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"

    elif config["dataset"] == "anuran":
        data = pd.read_csv('../data/Frogs_MFCCs.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x.startswith("MFCCs_")]
        categorical_features = [
            'Family',
            'Genus',
            'Species'
        ]
        integer_features = []
        ClfTarget = "Species"    
        print(continuous_features)
    elif config["dataset"] == "banknote":
        data = pd.read_csv('../data/data_banknote_authentication.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "variance", "skewness", "curtosis", "entropy"
        ]
        categorical_features = [
            'class',
        ]
        integer_features = []
        ClfTarget = "class"
        
    elif config["dataset"] == "breast":
        data = pd.read_csv('../data/wdbc.data', header=None)
        data = data.drop(columns=[0]) # drop ID number
        columns = ["Diagnosis"]
        common_cols = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal dimension",
        ]
        columns += [f"{x}1" for x in common_cols]
        columns += [f"{x}2" for x in common_cols]
        columns += [f"{x}3" for x in common_cols]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = []
        continuous_features += [f"{x}1" for x in common_cols]
        continuous_features += [f"{x}2" for x in common_cols]
        continuous_features += [f"{x}3" for x in common_cols]
        categorical_features = [
            "Diagnosis"
        ]
        integer_features = []
        ClfTarget = "Diagnosis"
        
    elif config["dataset"] == "concrete":
        data = pd.read_excel('../data/Concrete_Data.xls')
        columns = [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age",
            "Concrete compressive strength"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Age")
        continuous_features = columns
        categorical_features = [
            "Age",
        ]
        integer_features = []
        ClfTarget = "Age"
        
    elif config["dataset"] == "kings":
        data = pd.read_csv('../data/kc_house_data.csv')
        
        continuous_features = [
            'price', 
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15',
        ]
        categorical_features = [
            'bedrooms',
            'bathrooms',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade', 
        ]
        integer_features = [
            'price',
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'sqft_living15',
            'sqft_lot15',
        ]
        ClfTarget = "grade"
        
    elif config["dataset"] == "letter":
        data = pd.read_csv('../data/letter-recognition.data', header=None)
        columns = [
            "lettr",
            "x-box",
            "y-box",
            "width",
            "high",
            "onpix",
            "x-bar",
            "y-bar",
            "x2bar",
            "y2bar",
            "xybar",
            "x2ybr",
            "xy2br",
            "x-ege",
            "xegvy",
            "y-ege",
            "yegvx",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("lettr")
        continuous_features = columns
        categorical_features = [
            "lettr"
        ]
        integer_features = columns
        ClfTarget = "lettr"
        
    elif config["dataset"] == "loan":
        data = pd.read_csv('../data/Bank_Personal_Loan_Modelling.csv')
        
        continuous_features = [
            'Age',
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        categorical_features = [
            'Family',
            'Personal Loan',
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        integer_features = [
            'Age',
            'Experience',
            'Income', 
            'Mortgage'
        ]
        data = data[continuous_features + categorical_features]
        data = data.dropna()
        ClfTarget = "Personal Loan"
        
    elif config["dataset"] == "redwine":
        data = pd.read_csv('../data/winequality-red.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif config["dataset"] == "shoppers":
        data = pd.read_csv('../data/online_shoppers_intention.csv')
        
        assert data.isna().sum().sum() == 0

        continuous_features = [
            'Administrative_Duration',   
            'Informational_Duration',      
            'ProductRelated_Duration',     
            'BounceRates',             
            'ExitRates',                   
            'PageValues',                
            'SpecialDay',                
            'Administrative',    
            'Informational',     
            'ProductRelated',      
        ]

        categorical_features = [
            'Month',               
            'VisitorType',         
            'Weekend',          
            'OperatingSystems',    
            'Browser',            
            'Region',           
            'TrafficType',        
            "Revenue"
        ]

        integer_features = [
            'Administrative',    
            'Informational',      
            'ProductRelated',     
        ]

        ClfTarget = "Revenue"

    elif config["dataset"] == "whitewine":
        data = pd.read_csv('../data/winequality-white.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    return data, continuous_features, categorical_features, integer_features, ClfTarget

# %%